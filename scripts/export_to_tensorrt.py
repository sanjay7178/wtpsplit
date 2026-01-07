"""
Script to export SaT models to NVIDIA Triton Inference Server (TensorRT Backend).

This script:
1. Exports the model to generic ONNX format.
2. Compiles the ONNX model into a TensorRT Engine (.plan).
3. Generates the necessary config.pbtxt for Triton with "tensorrt_plan" backend.

Usage:
    python scripts/export_to_triton_trt.py \
        --model_name_or_path segment-any-text/sat-3l-sm \
        --output_dir triton_models/sat-3l-sm \
        --triton_model_name sat_3l_sm \
        --max_batch_size 32
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import adapters  # noqa
import torch
import onnx
import tensorrt as trt
from adapters.models import MODEL_MIXIN_MAPPING  # noqa
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin  # noqa
from huggingface_hub import hf_hub_download
from transformers import AutoModelForTokenClassification, HfArgumentParser

import wtpsplit  # noqa
import wtpsplit.models  # noqa
from wtpsplit.utils import Constants

MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin

# Setup TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

@dataclass
class Args:
    model_name_or_path: str = "segment-any-text/sat-3l-sm"
    output_dir: str = "triton_models/sat-3l-sm"
    triton_model_name: str = "sat_3l_sm"
    triton_version: str = "1"
    device: str = "cuda"
    max_batch_size: int = 32
    max_seq_len: int = 512
    use_lora: bool = False
    lora_path: str = None
    style_or_domain: str = "ud"
    language: str = "en"
    fp16: bool = True  # Enable FP16 for T4 optimization


def build_tensorrt_engine(onnx_file_path, engine_file_path, args):
    """Compiles ONNX model to TensorRT Engine (.plan)"""
    print(f"Building TensorRT Engine from {onnx_file_path}...")
    
    builder = trt.Builder(TRT_LOGGER)
    
    # EXPLICIT_BATCH is required for dynamic shapes in TRT 8+
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    config = builder.create_builder_config()
    
    # Enable FP16 if requested (Recommended for T4)
    if args.fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 Optimization Enabled.")
        else:
            print("Warning: FP16 requested but not supported on this platform. Falling back to FP32.")

    # Set Memory Pool (Workspace size) - Adjust based on available GPU memory
    # 2GB is usually sufficient for BERT-like models
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)

    # Parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)

    # Define Optimization Profile for Dynamic Shapes
    # We must tell TRT the Min, Opt, and Max dimensions for inputs
    profile = builder.create_optimization_profile()
    
    # Input: input_ids [batch, sequence]
    profile.set_shape(
        "input_ids",
        (1, 1),                        # Min: Batch 1, Seq 1
        (8, 256),                      # Opt: Batch 8, Seq 256 (Target optimization point)
        (args.max_batch_size, args.max_seq_len) # Max: Hard limit
    )
    
    # Input: attention_mask [batch, sequence]
    profile.set_shape(
        "attention_mask",
        (1, 1),
        (8, 256),
        (args.max_batch_size, args.max_seq_len)
    )
    
    config.add_optimization_profile(profile)

    # Build Engine
    print("Compiling engine... (this may take a few minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Engine build failed.")
        sys.exit(1)
        
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT Engine saved to {engine_file_path}")


def create_triton_config(
    model_name: str,
    max_batch_size: int,
    input_shape: list,
    output_shape: list,
) -> str:
    """Create a Triton config.pbtxt for TensorRT backend."""
    config = f"""name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [{input_shape}]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [{input_shape}]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP16
    dims: [{output_shape}]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

# TensorRT handles batching internally via the profile, 
# but Triton Dynamic Batching helps group requests before sending to TRT.
dynamic_batching {{
  preferred_batch_size: [1, 2, 4, 8, 16, {max_batch_size}]
  max_queue_delay_microseconds: 100
}}
"""
    return config


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()

    output_dir = Path(args.output_dir)
    model_repo_dir = output_dir / args.triton_version
    model_repo_dir.mkdir(exist_ok=True, parents=True)

    print(f"Exporting model to Triton TensorRT format at {output_dir}")

    # 1. Load PyTorch Model
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, force_download=False)
    model = model.to(args.device)

    # Fetch config.json from huggingface hub
    hf_hub_download(
        repo_id=args.model_name_or_path,
        filename="config.json",
        local_dir=output_dir,
    )

    # LoRA SETUP
    if args.use_lora:
        model_type = model.config.model_type
        model.config.model_type = "xlm-roberta"
        adapters.init(model)
        model.config.model_type = model_type
        
        if not args.lora_path:
            for file in [
                "adapter_config.json",
                "head_config.json",
                "pytorch_adapter.bin",
                "pytorch_model_head.bin",
            ]:
                hf_hub_download(
                    repo_id=args.model_name_or_path,
                    subfolder=f"loras/{args.style_or_domain}/{args.language}",
                    filename=file,
                    local_dir=Constants.CACHE_DIR,
                )
            lora_load_path = str(Constants.CACHE_DIR / "loras" / args.style_or_domain / args.language)
        else:
            lora_load_path = args.lora_path

        print(f"Using LoRA weights from {lora_load_path}.")
        model.load_adapter(
            lora_load_path,
            set_active=True,
            with_head=True,
            load_as="sat-lora",
        )
        model.merge_adapter("sat-lora")
        print("LoRA setup done.")

    # Move to half precision for export if T4
    if args.fp16:
        model = model.half()

    # 2. Export to Intermediate ONNX
    # We do NOT use onnxruntime optimizer here because it injects Microsoft-specific ops
    # that TensorRT parser does not understand. We want standard ONNX ops.
    onnx_path = model_repo_dir / "model_intermediate.onnx"
    print(f"Exporting intermediate ONNX to {onnx_path}")
    
    # Generate dummy inputs (ensure int32 for TensorRT compatibility usually preferred)
    dummy_inputs = (
        torch.randint(0, model.config.vocab_size, (1, 512), dtype=torch.int32, device=args.device),
        torch.ones((1, 512), dtype=torch.int32, device=args.device),
    )

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        verbose=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17, 
        # dynamo=False,
    )
    
    # Downgrade IR version if necessary for the Parser (seldom needed for newer TRT, but safe)
    onnx_model_tmp = onnx.load(onnx_path)
    if onnx_model_tmp.ir_version > 9:
        onnx_model_tmp.ir_version = 9
        onnx.save(onnx_model_tmp, onnx_path)

    # 3. Build TensorRT Engine
    trt_plan_path = model_repo_dir / "model.plan"
    build_tensorrt_engine(str(onnx_path), str(trt_plan_path), args)

    # Clean up intermediate ONNX to save space
    # onnx_path.unlink() # Uncomment if you want to delete the .onnx file

    # 4. Create Triton config
    config_path = output_dir / "config.pbtxt"
    config_content = create_triton_config(
        model_name=args.triton_model_name,
        max_batch_size=args.max_batch_size,
        input_shape="-1",  # Matches the profile dimensions
        output_shape="-1, 1",
    )
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print(f"Created Triton config at {config_path}")
    print(f"\nSUCCESS: TensorRT-optimized Triton model ready!")