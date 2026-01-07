"""
BentoML service for wtpsplit sentence segmentation with ONNX GPU Runtime.

High-throughput production-ready serving for sentence segmentation.

Usage:
    # Start the server directly
    bentoml serve bentoml_service:WtpSplitService --port 8000

    # Or via serve_onnx_api.py with configuration
    python serve_onnx_api.py --bentoml --max-batch-size 64 --port 8000

    # Test single text
    curl -X POST "http://localhost:8000/split" \
        -H "Content-Type: application/json" \
        -d '{"request": {"text": "Hello world. This is a test."}}'

    # Test batch texts
    curl -X POST "http://localhost:8000/split" \
        -H "Content-Type: application/json" \
        -d '{"request": {"texts": ["First text.", "Second text."]}}'

    # Health check
    curl -X POST "http://localhost:8000/health"

    # Get statistics
    curl -X POST "http://localhost:8000/stats"

Environment Variables:
    WTPSPLIT_MODEL: Model name (default: segment-any-text/sat-3l-sm)
    WTPSPLIT_MAX_BATCH_SIZE: Max batch size (default: 64)
    WTPSPLIT_MAX_LATENCY_MS: Max batch wait time in ms (default: 10)
    WTPSPLIT_TIMEOUT: Request timeout in seconds (default: 300)
    WTPSPLIT_CONCURRENCY: Max concurrent requests (default: 64)
"""

import os
import time
import logging
from typing import List, Optional

import bentoml
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wtpsplit-bentoml")

# Configuration from environment variables
MODEL_NAME = os.environ.get("WTPSPLIT_MODEL", "segment-any-text/sat-3l-sm")
MAX_BATCH_SIZE = int(os.environ.get("WTPSPLIT_MAX_BATCH_SIZE", "64"))
MAX_LATENCY_MS = int(os.environ.get("WTPSPLIT_MAX_LATENCY_MS", "10"))
TIMEOUT = float(os.environ.get("WTPSPLIT_TIMEOUT", "300"))
CONCURRENCY = int(os.environ.get("WTPSPLIT_CONCURRENCY", "64"))


# -----------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------

class SplitRequest(BaseModel):
    """Request model for text segmentation."""
    text: Optional[str] = Field(None, description="Single text to segment")
    texts: Optional[List[str]] = Field(None, description="Multiple texts to segment (batch)")
    threshold: Optional[float] = Field(None, description="Segmentation threshold")
    stride: int = Field(64, description="Stride for sliding window")
    block_size: int = Field(512, description="Maximum block size for processing")
    strip_whitespace: bool = Field(False, description="Strip whitespace from sentences")


class SplitResponse(BaseModel):
    """Response model for text segmentation."""
    sentences: Optional[List[str]] = Field(None, description="Segmented sentences (single text)")
    batch_sentences: Optional[List[List[str]]] = Field(None, description="Segmented sentences (batch)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ProbabilityRequest(BaseModel):
    """Request model for probability prediction."""
    text: str = Field(..., description="Text to get split probabilities for")
    stride: int = Field(256, description="Stride for sliding window")
    block_size: int = Field(512, description="Maximum block size")


class ProbabilityResponse(BaseModel):
    """Response model for probability prediction."""
    probabilities: List[float] = Field(..., description="Split probability for each character")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchingConfig(BaseModel):
    """Batching configuration."""
    enabled: bool
    max_batch_size: int
    max_latency_ms: int


class ServiceConfig(BaseModel):
    """Service configuration."""
    max_batch_size: int
    max_latency_ms: int
    timeout: float
    concurrency: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_name: str
    backend: str
    gpu_available: bool
    batching: BatchingConfig


class StatsResponse(BaseModel):
    """Statistics response."""
    model_name: str
    backend: str
    gpu_available: bool
    requests_processed: int
    batches_processed: int
    average_batch_size: float
    config: ServiceConfig


# -----------------------------------------------------------------
# BentoML Service
# -----------------------------------------------------------------
@bentoml.service(
    name="wtpsplit-onnx",
    resources={"gpu": 1, "memory": "8Gi"},
    traffic={
        "timeout": TIMEOUT, 
        "concurrency": CONCURRENCY,
        "max_concurrency": 512,
    },
)
class WtpSplitService:
    
    def __init__(self):
        from wtpsplit import SaT
        import onnxruntime as ort
        
        self.model_name = MODEL_NAME
        self.max_batch_size = MAX_BATCH_SIZE
        self.max_latency_ms = MAX_LATENCY_MS
        
        logger.info("Initializing WtpSplitService...")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Max Batch Size: {self.max_batch_size}")
        logger.info(f"  Max Latency: {self.max_latency_ms}ms")
        logger.info(f"  Timeout: {TIMEOUT}s")
        logger.info(f"  Concurrency: {CONCURRENCY}")
        
        # Check GPU availability
        try:
            available_providers = ort.get_available_providers()
            self.gpu_available = "CUDAExecutionProvider" in available_providers
            logger.info(f"Available ONNX providers: {available_providers}")
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
            self.gpu_available = False
        
        # Optimized CUDA provider options for T4
        cuda_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'gpu_mem_limit': 14 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }
        
        self.ort_providers = [
            ('CUDAExecutionProvider', cuda_options),
            'CPUExecutionProvider'
        ]
        
        # Load model
        logger.info(f"Loading model: {self.model_name}")
        start_time = time.perf_counter()
        self.model = SaT(self.model_name, ort_providers=self.ort_providers)
        load_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Model loaded in {load_time:.2f}ms")
        
        # Extended warmup for CUDA kernel optimization
        warmup_size = min(self.max_batch_size, 32)
        logger.info(f"Warming up model with batch size {warmup_size}...")
        warmup_texts = [f"Warmup sentence {i}. Another sentence." for i in range(warmup_size)]
        for _ in range(10):  # More warmup iterations
            _ = list(self.model.split(warmup_texts))
        logger.info("Model warmup complete")
        
        # Stats tracking
        self._stats = {
            "requests_processed": 0,
            "batches_processed": 0,
            "total_batch_size": 0,
        }
        
        logger.info("WtpSplitService initialized successfully")
    

    # Add batchable endpoint for automatic batching
    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def split_batched(self, texts: List[str]) -> List[List[str]]:
        """Auto-batched endpoint - use this for high throughput."""
        return list(self.model.split(texts))

    def _split_texts(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
        stride: int = 64,
        block_size: int = 512,
        strip_whitespace: bool = False,
    ) -> List[List[str]]:
        """Internal method to split texts."""
        batch_size = len(texts)
        logger.debug(f"Processing batch of {batch_size} texts")
        
        results = list(self.model.split(
            texts,
            threshold=threshold,
            stride=stride,
            block_size=block_size,
            strip_whitespace=strip_whitespace,
        ))
        
        # Update stats
        self._stats["batches_processed"] += 1
        self._stats["requests_processed"] += batch_size
        self._stats["total_batch_size"] += batch_size
        
        return results
    
    @bentoml.api()
    def split(self, request: SplitRequest) -> SplitResponse:
        """
        Split single text or batch of texts into sentences.
        
        For single texts, the request is automatically batched with
        other concurrent requests via the split_batch method.
        """
        start_time = time.perf_counter()
        
        if request.text is None and request.texts is None:
            raise ValueError("Either 'text' or 'texts' must be provided")
        
        if request.text is not None and request.texts is not None:
            raise ValueError("Provide either 'text' or 'texts', not both")
        
        if request.text is not None:
            # Single text
            result = self._split_texts(
                [request.text],
                threshold=request.threshold,
                stride=request.stride,
                block_size=request.block_size,
                strip_whitespace=request.strip_whitespace,
            )
            processing_time = (time.perf_counter() - start_time) * 1000
            return SplitResponse(
                sentences=result[0],
                processing_time_ms=processing_time,
            )
        else:
            # Batch of texts
            results = self._split_texts(
                request.texts,
                threshold=request.threshold,
                stride=request.stride,
                block_size=request.block_size,
                strip_whitespace=request.strip_whitespace,
            )
            processing_time = (time.perf_counter() - start_time) * 1000
            return SplitResponse(
                batch_sentences=results,
                processing_time_ms=processing_time,
            )
    
    @bentoml.api()
    def probabilities(self, request: ProbabilityRequest) -> ProbabilityResponse:
        """
        Get split probabilities for a text.
        
        Useful for custom post-processing or visualization.
        """
        start_time = time.perf_counter()
        
        probs = self.model.predict_proba(
            request.text,
            stride=request.stride,
            block_size=request.block_size,
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000
        return ProbabilityResponse(
            probabilities=probs.tolist(),
            processing_time_ms=processing_time,
        )
    
    @bentoml.api()
    def health(self) -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_name=self.model_name,
            backend="onnxruntime-gpu (BentoML)",
            gpu_available=self.gpu_available,
            batching=BatchingConfig(
                enabled=True,
                max_batch_size=self.max_batch_size,
                max_latency_ms=self.max_latency_ms,
            ),
        )
    
    @bentoml.api()
    def stats(self) -> StatsResponse:
        """Get service statistics."""
        avg_batch_size = (
            self._stats["total_batch_size"] / self._stats["batches_processed"]
            if self._stats["batches_processed"] > 0 else 0
        )
        return StatsResponse(
            model_name=self.model_name,
            backend="onnxruntime-gpu (BentoML)",
            gpu_available=self.gpu_available,
            requests_processed=self._stats["requests_processed"],
            batches_processed=self._stats["batches_processed"],
            average_batch_size=round(avg_batch_size, 2),
            config=ServiceConfig(
                max_batch_size=self.max_batch_size,
                max_latency_ms=self.max_latency_ms,
                timeout=TIMEOUT,
                concurrency=CONCURRENCY,
            ),
        )

