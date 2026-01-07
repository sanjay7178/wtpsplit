"""
Ray Serve + FastAPI + BentoML API for wtpsplit sentence segmentation using ONNX GPU Runtime (CUDA).

This provides multiple REST API alternatives to Triton Server for serving wtpsplit models:
- Ray Serve: Scalable deployment with autoscaling
- Standalone: High-throughput with ThreadPool or AutoBatch
- BentoML: Production-ready serving with adaptive batching

Usage:
    # Ray Serve mode (default)
    python serve_onnx_api.py

    # Standalone ThreadPool mode
    python serve_onnx_api.py --standalone --num-workers 8

    # Standalone AutoBatch mode
    python serve_onnx_api.py --standalone --auto-batch --max-batch-size 32

    # BentoML mode (high-throughput with adaptive batching)
    python serve_onnx_api.py --bentoml

    # BentoML with custom settings
    python serve_onnx_api.py --bentoml --max-batch-size 64 --max-batch-wait-ms 10

    # Test the API
    curl -X POST "http://localhost:8000/split" \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world. This is a test."}'

Requirements:
    pip install "ray[serve]" fastapi uvicorn onnxruntime-gpu bentoml
"""

import argparse
import logging
import time
from typing import List, Optional
from contextlib import asynccontextmanager

import ray
from ray import serve
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wtpsplit-api")


# -----------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------

class SplitRequest(BaseModel):
    """Request model for text segmentation."""
    text: Optional[str] = Field(None, description="Single text to segment")
    texts: Optional[List[str]] = Field(None, description="Multiple texts to segment (batch)")
    threshold: Optional[float] = Field(None, description="Segmentation threshold (default: model-specific)")
    stride: int = Field(64, description="Stride for sliding window")
    block_size: int = Field(512, description="Maximum block size for processing")
    strip_whitespace: bool = Field(False, description="Strip whitespace from sentences")
    do_paragraph_segmentation: bool = Field(False, description="Also segment into paragraphs")


class SplitResponse(BaseModel):
    """Response model for text segmentation."""
    sentences: Optional[List[str]] = Field(None, description="Segmented sentences (single text)")
    batch_sentences: Optional[List[List[str]]] = Field(None, description="Segmented sentences (batch)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ProbabilityRequest(BaseModel):
    """Request model for probability prediction."""
    text: str = Field(..., description="Text to get split probabilities for")
    stride: int = Field(256, description="Stride for sliding window")
    block_size: int = Field(512, description="Maximum block size for processing")


class ProbabilityResponse(BaseModel):
    """Response model for probability prediction."""
    probabilities: List[float] = Field(..., description="Split probability for each character")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_name: str
    backend: str
    gpu_available: bool


class BatchSplitRequest(BaseModel):
    """Request model for batch text segmentation."""
    texts: List[str] = Field(..., description="List of texts to segment")
    threshold: Optional[float] = Field(None, description="Segmentation threshold")
    stride: int = Field(64, description="Stride for sliding window")
    block_size: int = Field(512, description="Maximum block size for processing")
    strip_whitespace: bool = Field(False, description="Strip whitespace from sentences")


class BatchSplitResponse(BaseModel):
    """Response model for batch text segmentation."""
    results: List[List[str]] = Field(..., description="Segmented sentences for each text")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    texts_processed: int = Field(..., description="Number of texts processed")


# -----------------------------------------------------------------
# Global state for lifespan management
# -----------------------------------------------------------------

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    logger.info("Starting wtpsplit ONNX API...")
    yield
    logger.info("Shutting down wtpsplit ONNX API...")


# -----------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------

app = FastAPI(
    title="wtpsplit ONNX API",
    description="REST API for sentence segmentation using wtpsplit with ONNX GPU Runtime",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------
# Ray Serve Deployment
# -----------------------------------------------------------------

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 10,
    },
)
@serve.ingress(app)
class WtpSplitONNXDeployment:
    """
    Ray Serve deployment for wtpsplit with ONNX GPU Runtime.
    
    This deployment loads the model once and handles inference requests.
    Uses CUDA execution provider for GPU acceleration.
    """
    
    def __init__(
        self,
        model_name: str = "segment-any-text/sat-3l-sm",
        ort_providers: List[str] = None,
    ):
        """
        Initialize the deployment with the specified model.
        
        Args:
            model_name: HuggingFace model name or local path
            ort_providers: ONNX Runtime execution providers (default: CUDA + CPU)
        """
        from wtpsplit import SaT
        
        self.model_name = model_name
        self.ort_providers = ort_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        logger.info(f"Loading model: {model_name}")
        logger.info(f"ONNX Runtime providers: {self.ort_providers}")
        
        # Check GPU availability
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            self.gpu_available = "CUDAExecutionProvider" in available_providers
            logger.info(f"Available ONNX providers: {available_providers}")
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
            self.gpu_available = False
        
        # Load the model
        start_time = time.perf_counter()
        self.model = SaT(
            model_name,
            ort_providers=self.ort_providers,
        )
        load_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Model loaded in {load_time:.2f} ms")
        
        # Warmup
        logger.info("Warming up model...")
        for _ in range(5):
            _ = self.model.split("This is a warmup sentence. Here is another one.")
        logger.info("Model warmup complete")
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_name=self.model_name,
            backend="onnxruntime-gpu",
            gpu_available=self.gpu_available,
        )
    
    @app.get("/")
    async def root(self):
        """Root endpoint with API info."""
        return {
            "service": "wtpsplit ONNX API",
            "model": self.model_name,
            "backend": "onnxruntime-gpu",
            "endpoints": {
                "/split": "POST - Segment text into sentences",
                "/split/batch": "POST - Batch segment multiple texts",
                "/probabilities": "POST - Get split probabilities",
                "/health": "GET - Health check",
            }
        }
    
    @app.post("/split", response_model=SplitResponse)
    async def split_text(self, request: SplitRequest) -> SplitResponse:
        """
        Segment text into sentences.
        
        Accepts either a single text or a batch of texts.
        """
        start_time = time.perf_counter()
        
        # Validate input
        if request.text is None and request.texts is None:
            raise HTTPException(status_code=400, detail="Either 'text' or 'texts' must be provided")
        
        if request.text is not None and request.texts is not None:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts', not both")
        
        try:
            if request.text is not None:
                # Single text
                sentences = self.model.split(
                    request.text,
                    threshold=request.threshold,
                    stride=request.stride,
                    block_size=request.block_size,
                    strip_whitespace=request.strip_whitespace,
                    do_paragraph_segmentation=request.do_paragraph_segmentation,
                )
                processing_time = (time.perf_counter() - start_time) * 1000
                return SplitResponse(
                    sentences=sentences,
                    processing_time_ms=processing_time,
                )
            else:
                # Batch processing
                results = list(self.model.split(
                    request.texts,
                    threshold=request.threshold,
                    stride=request.stride,
                    block_size=request.block_size,
                    strip_whitespace=request.strip_whitespace,
                    do_paragraph_segmentation=request.do_paragraph_segmentation,
                ))
                processing_time = (time.perf_counter() - start_time) * 1000
                return SplitResponse(
                    batch_sentences=results,
                    processing_time_ms=processing_time,
                )
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/split/batch", response_model=BatchSplitResponse)
    async def split_batch(self, request: BatchSplitRequest) -> BatchSplitResponse:
        """
        Batch segment multiple texts into sentences.
        
        More efficient for processing multiple texts at once.
        """
        start_time = time.perf_counter()
        
        if not request.texts:
            raise HTTPException(status_code=400, detail="'texts' list cannot be empty")
        
        try:
            results = list(self.model.split(
                request.texts,
                threshold=request.threshold,
                stride=request.stride,
                block_size=request.block_size,
                strip_whitespace=request.strip_whitespace,
            ))
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return BatchSplitResponse(
                results=results,
                processing_time_ms=processing_time,
                texts_processed=len(request.texts),
            )
        except Exception as e:
            logger.error(f"Error processing batch request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/probabilities", response_model=ProbabilityResponse)
    async def get_probabilities(self, request: ProbabilityRequest) -> ProbabilityResponse:
        """
        Get split probabilities for each character in the text.
        
        Useful for custom post-processing or visualization.
        """
        start_time = time.perf_counter()
        
        try:
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
        except Exception as e:
            logger.error(f"Error getting probabilities: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------
# Standalone FastAPI App (without Ray Serve)
# -----------------------------------------------------------------

def create_standalone_app(
    model_name: str = "segment-any-text/sat-3l-sm",
    ort_providers: List[str] = None,
    num_workers: int = 4,
    enable_batching: bool = False,
    max_batch_size: int = 32,
    max_batch_wait_ms: float = 5.0,
) -> FastAPI:
    """
    Create a standalone FastAPI app without Ray Serve.
    
    Uses a ThreadPoolExecutor for high-throughput concurrent inference.
    Optionally enables request auto-batching for maximum GPU efficiency.
    
    Args:
        model_name: HuggingFace model name or local path
        ort_providers: ONNX Runtime execution providers
        num_workers: Number of thread pool workers for inference
        enable_batching: Enable request auto-batching (collects requests into batches)
        max_batch_size: Maximum batch size for auto-batching
        max_batch_wait_ms: Maximum time to wait for batch to fill (milliseconds)
    """
    from wtpsplit import SaT
    from concurrent.futures import ThreadPoolExecutor
    import asyncio
    import threading
    from dataclasses import dataclass
    from collections import deque
    
    ort_providers = ort_providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    # -----------------------------------------------------------------
    # Auto-Batching Infrastructure
    # -----------------------------------------------------------------
    
    @dataclass
    class BatchRequest:
        """A single request waiting to be batched."""
        text: str
        threshold: Optional[float]
        stride: int
        block_size: int
        strip_whitespace: bool
        future: asyncio.Future
        loop: asyncio.AbstractEventLoop
    
    class AutoBatcher:
        """
        Collects incoming requests and batches them for efficient GPU inference.
        
        Similar to Triton's dynamic batching but implemented in Python.
        """
        
        def __init__(
            self,
            model,
            max_batch_size: int = 32,
            max_wait_ms: float = 5.0,
        ):
            self.model = model
            self.max_batch_size = max_batch_size
            self.max_wait_ms = max_wait_ms
            self.queue: deque[BatchRequest] = deque()
            self.lock = threading.Lock()
            self.condition = threading.Condition(self.lock)
            self.running = True
            self.worker_thread = None
            self.stats = {
                "batches_processed": 0,
                "requests_processed": 0,
                "total_batch_size": 0,
            }
        
        def start(self):
            """Start the background batch processing worker."""
            self.worker_thread = threading.Thread(
                target=self._batch_worker,
                daemon=True,
                name="batch_worker"
            )
            self.worker_thread.start()
            logger.info(f"AutoBatcher started (max_batch={self.max_batch_size}, max_wait={self.max_wait_ms}ms)")
        
        def stop(self):
            """Stop the background worker."""
            self.running = False
            with self.condition:
                self.condition.notify_all()
            if self.worker_thread:
                self.worker_thread.join(timeout=5.0)
            logger.info("AutoBatcher stopped")
        
        def submit(self, request: BatchRequest):
            """Submit a request for batched processing."""
            with self.condition:
                self.queue.append(request)
                self.condition.notify()
        
        def _batch_worker(self):
            """Background worker that collects and processes batches."""
            while self.running:
                batch = []
                
                with self.condition:
                    # Wait for at least one request
                    while self.running and len(self.queue) == 0:
                        self.condition.wait()
                    
                    if not self.running:
                        break
                    
                    # Collect requests up to max_batch_size or until timeout
                    deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)
                    
                    while len(batch) < self.max_batch_size:
                        # Check if we have requests
                        if self.queue:
                            batch.append(self.queue.popleft())
                        else:
                            # Wait for more requests or timeout
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                break
                            self.condition.wait(timeout=remaining)
                            
                            if not self.running:
                                break
                
                if not batch:
                    continue
                
                # Process the batch
                try:
                    self._process_batch(batch)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
                    # Set error on all futures
                    for req in batch:
                        if not req.future.done():
                            req.loop.call_soon_threadsafe(
                                req.future.set_exception, e
                            )
        
        def _process_batch(self, batch: List[BatchRequest]):
            """Process a batch of requests together."""
            texts = [req.text for req in batch]
            
            # Use first request's parameters (could be made smarter)
            first_req = batch[0]
            
            try:
                # Run batch inference
                results = list(self.model.split(
                    texts,
                    threshold=first_req.threshold,
                    stride=first_req.stride,
                    block_size=first_req.block_size,
                    strip_whitespace=first_req.strip_whitespace,
                ))
                
                # Update stats
                self.stats["batches_processed"] += 1
                self.stats["requests_processed"] += len(batch)
                self.stats["total_batch_size"] += len(batch)
                
                # Distribute results to waiting requests
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.loop.call_soon_threadsafe(
                            req.future.set_result, result
                        )
                        
            except Exception as e:
                # Set error on all futures
                for req in batch:
                    if not req.future.done():
                        req.loop.call_soon_threadsafe(
                            req.future.set_exception, e
                        )
                raise
        
        def get_stats(self) -> dict:
            """Get batching statistics."""
            avg_batch_size = (
                self.stats["total_batch_size"] / self.stats["batches_processed"]
                if self.stats["batches_processed"] > 0 else 0
            )
            return {
                "batches_processed": self.stats["batches_processed"],
                "requests_processed": self.stats["requests_processed"],
                "average_batch_size": round(avg_batch_size, 2),
                "queue_size": len(self.queue),
            }
    
    # -----------------------------------------------------------------
    # FastAPI App
    # -----------------------------------------------------------------
    
    mode_name = "AutoBatch" if enable_batching else "ThreadPool"
    standalone_app = FastAPI(
        title=f"wtpsplit ONNX API (Standalone - {mode_name})",
        description="High-throughput REST API for sentence segmentation using wtpsplit with ONNX GPU Runtime",
        version="1.0.0",
    )
    
    standalone_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Shared state
    model_state = {
        "model": None,
        "model_name": model_name,
        "gpu_available": False,
        "executor": None,
        "num_workers": num_workers,
        "batcher": None,
        "enable_batching": enable_batching,
        "max_batch_size": max_batch_size,
        "max_batch_wait_ms": max_batch_wait_ms,
    }
    
    @standalone_app.on_event("startup")
    async def startup():
        logger.info(f"Loading model: {model_name}")
        
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            model_state["gpu_available"] = "CUDAExecutionProvider" in available_providers
            logger.info(f"Available providers: {available_providers}")
        except Exception:
            pass
        
        # Load model
        model_state["model"] = SaT(model_name, ort_providers=ort_providers)
        
        if enable_batching:
            # Auto-batching mode
            logger.info(f"Initializing AutoBatcher (max_batch={max_batch_size}, max_wait={max_batch_wait_ms}ms)")
            model_state["batcher"] = AutoBatcher(
                model=model_state["model"],
                max_batch_size=max_batch_size,
                max_wait_ms=max_batch_wait_ms,
            )
            model_state["batcher"].start()
            
            # Warmup with batch
            logger.info("Warming up model with batch inference...")
            warmup_texts = [f"Warmup sentence {i}. Another one." for i in range(max_batch_size)]
            _ = list(model_state["model"].split(warmup_texts))
            logger.info(f"Model loaded with AutoBatcher (max_batch={max_batch_size})")
        else:
            # ThreadPool mode
            logger.info(f"Initializing ThreadPoolExecutor with {num_workers} workers")
            model_state["executor"] = ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix="inference_worker"
            )
            
            # Warmup - run in thread pool to warm up all workers
            logger.info("Warming up model across all workers...")
            warmup_futures = []
            for i in range(num_workers * 2):
                future = model_state["executor"].submit(
                    model_state["model"].split,
                    f"Warmup sentence {i}. Another sentence for worker warmup."
                )
                warmup_futures.append(future)
            
            # Wait for all warmup to complete
            for future in warmup_futures:
                future.result()
            
            logger.info(f"Model loaded and warmed up with {num_workers} workers")
    
    @standalone_app.on_event("shutdown")
    async def shutdown():
        if model_state["batcher"]:
            logger.info("Shutting down AutoBatcher...")
            model_state["batcher"].stop()
            logger.info("AutoBatcher shut down")
        if model_state["executor"]:
            logger.info("Shutting down thread pool executor...")
            model_state["executor"].shutdown(wait=True)
            logger.info("Thread pool executor shut down")
    
    @standalone_app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            model_name=model_state["model_name"],
            backend="onnxruntime-gpu",
            gpu_available=model_state["gpu_available"],
        )
    
    @standalone_app.get("/")
    async def root():
        mode = "AutoBatch" if model_state["enable_batching"] else "ThreadPool"
        response = {
            "service": f"wtpsplit ONNX API (Standalone - {mode})",
            "model": model_state["model_name"],
            "backend": "onnxruntime-gpu",
            "gpu_available": model_state["gpu_available"],
            "mode": mode,
        }
        if model_state["enable_batching"]:
            response["max_batch_size"] = model_state["max_batch_size"]
            response["max_batch_wait_ms"] = model_state["max_batch_wait_ms"]
        else:
            response["num_workers"] = model_state["num_workers"]
        return response
    
    @standalone_app.get("/stats")
    async def get_stats():
        """Get server statistics."""
        stats = {
            "model_name": model_state["model_name"],
            "gpu_available": model_state["gpu_available"],
            "mode": "AutoBatch" if model_state["enable_batching"] else "ThreadPool",
        }
        
        if model_state["enable_batching"] and model_state["batcher"]:
            stats["batching"] = model_state["batcher"].get_stats()
            stats["max_batch_size"] = model_state["max_batch_size"]
            stats["max_batch_wait_ms"] = model_state["max_batch_wait_ms"]
        else:
            executor = model_state["executor"]
            stats["num_workers"] = model_state["num_workers"]
            stats["executor_threads"] = executor._max_workers if executor else 0
        
        return stats
    
    def _do_split(model, text, threshold, stride, block_size, strip_whitespace, do_paragraph_segmentation):
        """Synchronous split function to run in thread pool."""
        return model.split(
            text,
            threshold=threshold,
            stride=stride,
            block_size=block_size,
            strip_whitespace=strip_whitespace,
            do_paragraph_segmentation=do_paragraph_segmentation,
        )
    
    def _do_split_batch(model, texts, threshold, stride, block_size, strip_whitespace):
        """Synchronous batch split function to run in thread pool."""
        return list(model.split(
            texts,
            threshold=threshold,
            stride=stride,
            block_size=block_size,
            strip_whitespace=strip_whitespace,
        ))
    
    def _do_predict_proba(model, text, stride, block_size):
        """Synchronous predict_proba function to run in thread pool."""
        return model.predict_proba(text, stride=stride, block_size=block_size)
    
    @standalone_app.post("/split", response_model=SplitResponse)
    async def split_text(request: SplitRequest):
        start_time = time.perf_counter()
        model = model_state["model"]
        batcher = model_state["batcher"]
        executor = model_state["executor"]
        loop = asyncio.get_event_loop()
        
        if request.text is None and request.texts is None:
            raise HTTPException(status_code=400, detail="Either 'text' or 'texts' must be provided")
        
        if request.text is not None and request.texts is not None:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'texts', not both")
        
        try:
            if request.text is not None:
                if batcher:
                    # Auto-batching mode: submit to batcher queue
                    future = loop.create_future()
                    batch_req = BatchRequest(
                        text=request.text,
                        threshold=request.threshold,
                        stride=request.stride,
                        block_size=request.block_size,
                        strip_whitespace=request.strip_whitespace,
                        future=future,
                        loop=loop,
                    )
                    batcher.submit(batch_req)
                    sentences = await future
                else:
                    # ThreadPool mode: run in executor
                    sentences = await loop.run_in_executor(
                        executor,
                        _do_split,
                        model,
                        request.text,
                        request.threshold,
                        request.stride,
                        request.block_size,
                        request.strip_whitespace,
                        request.do_paragraph_segmentation,
                    )
                processing_time = (time.perf_counter() - start_time) * 1000
                return SplitResponse(sentences=sentences, processing_time_ms=processing_time)
            else:
                # Batch input - always use direct batch inference (not auto-batching queue)
                if executor:
                    results = await loop.run_in_executor(
                        executor,
                        _do_split_batch,
                        model,
                        request.texts,
                        request.threshold,
                        request.stride,
                        request.block_size,
                        request.strip_whitespace,
                    )
                else:
                    # Direct call for batching mode
                    results = list(model.split(
                        request.texts,
                        threshold=request.threshold,
                        stride=request.stride,
                        block_size=request.block_size,
                        strip_whitespace=request.strip_whitespace,
                    ))
                processing_time = (time.perf_counter() - start_time) * 1000
                return SplitResponse(batch_sentences=results, processing_time_ms=processing_time)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @standalone_app.post("/split/batch", response_model=BatchSplitResponse)
    async def split_batch(request: BatchSplitRequest):
        start_time = time.perf_counter()
        model = model_state["model"]
        executor = model_state["executor"]
        loop = asyncio.get_event_loop()
        
        if not request.texts:
            raise HTTPException(status_code=400, detail="'texts' list cannot be empty")
        
        try:
            # Batch endpoint always uses direct batch inference
            if executor:
                results = await loop.run_in_executor(
                    executor,
                    _do_split_batch,
                    model,
                    request.texts,
                    request.threshold,
                    request.stride,
                    request.block_size,
                    request.strip_whitespace,
                )
            else:
                # Direct call for batching mode
                results = list(model.split(
                    request.texts,
                    threshold=request.threshold,
                    stride=request.stride,
                    block_size=request.block_size,
                    strip_whitespace=request.strip_whitespace,
                ))
            processing_time = (time.perf_counter() - start_time) * 1000
            return BatchSplitResponse(
                results=results,
                processing_time_ms=processing_time,
                texts_processed=len(request.texts),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @standalone_app.post("/probabilities", response_model=ProbabilityResponse)
    async def get_probabilities(request: ProbabilityRequest):
        start_time = time.perf_counter()
        model = model_state["model"]
        executor = model_state["executor"]
        loop = asyncio.get_event_loop()
        
        try:
            # Run inference in thread pool
            probs = await loop.run_in_executor(
                executor,
                _do_predict_proba,
                model,
                request.text,
                request.stride,
                request.block_size,
            )
            processing_time = (time.perf_counter() - start_time) * 1000
            return ProbabilityResponse(probabilities=probs.tolist(), processing_time_ms=processing_time)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return standalone_app


# -----------------------------------------------------------------
# BentoML Service Implementation
# -----------------------------------------------------------------

def start_bentoml(
    model_name: str = "segment-any-text/sat-3l-sm",
    host: str = "0.0.0.0",
    port: int = 8000,
    max_batch_size: int = 32,
    max_batch_wait_ms: float = 10.0,
    api_workers: int = 1,
    timeout: float = 300.0,
    max_concurrency: int = 32,
):
    """
    Start the BentoML server with adaptive batching for high throughput.
    
    Args:
        model_name: HuggingFace model name or local path
        host: Host to bind to
        port: Port to bind to
        max_batch_size: Maximum batch size for adaptive batching
        max_batch_wait_ms: Maximum latency to wait for batch formation
        api_workers: Number of API server workers (use 1 for GPU)
        timeout: Request timeout in seconds
        max_concurrency: Maximum concurrent requests per worker
    """
    import importlib.util
    if importlib.util.find_spec("bentoml") is None:
        logger.error("BentoML not installed. Run: pip install bentoml")
        return
    
    import subprocess
    import sys
    import os
    
    logger.info("=" * 60)
    logger.info(" Starting wtpsplit ONNX API (BentoML)")
    logger.info("=" * 60)
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Host: {host}:{port}")
    logger.info(f"  Max Batch Size: {max_batch_size}")
    logger.info(f"  Max Batch Wait: {max_batch_wait_ms} ms")
    logger.info(f"  API Workers: {api_workers}")
    logger.info(f"  Timeout: {timeout}s")
    logger.info(f"  Max Concurrency: {max_concurrency}")
    logger.info("=" * 60)
    
    # Set environment variables for the BentoML service
    env = os.environ.copy()
    env["WTPSPLIT_MODEL"] = model_name
    env["WTPSPLIT_MAX_BATCH_SIZE"] = str(max_batch_size)
    env["WTPSPLIT_MAX_LATENCY_MS"] = str(int(max_batch_wait_ms))
    env["WTPSPLIT_TIMEOUT"] = str(timeout)
    env["WTPSPLIT_CONCURRENCY"] = str(max_concurrency)
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    service_file = os.path.join(script_dir, "bentoml_service.py")
    
    if not os.path.exists(service_file):
        logger.error(f"BentoML service file not found: {service_file}")
        return
    
    # Build the bentoml serve command
    cmd = [
        sys.executable, "-m", "bentoml", "serve",
        "bentoml_service:WtpSplitService",
        "--host", host,
        "--port", str(port),
        "--api-workers", str(api_workers),
    ]
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, cwd=script_dir, check=True)
    except KeyboardInterrupt:
        logger.info("Shutting down BentoML server...")
    except subprocess.CalledProcessError as e:
        logger.error(f"BentoML server failed: {e}")


# -----------------------------------------------------------------
# Main Entry Points
# -----------------------------------------------------------------

def start_ray_serve(
    model_name: str = "segment-any-text/sat-3l-sm",
    host: str = "0.0.0.0",
    port: int = 8000,
    num_replicas: int = 1,
):
    """Start the Ray Serve deployment."""
    ray.init(ignore_reinit_error=True)
    
    # Configure the deployment
    deployment = WtpSplitONNXDeployment.options(
        num_replicas=num_replicas,
    ).bind(model_name=model_name)
    
    # Start serving
    serve.run(deployment)
    logger.info(f"Ray Serve started on http://{host}:{port}")
    
    # Keep the server running
    while True:
        time.sleep(1)


def start_standalone(
    model_name: str = "segment-any-text/sat-3l-sm",
    host: str = "0.0.0.0",
    port: int = 8000,
    num_workers: int = 4,
    uvicorn_workers: int = 1,
    enable_batching: bool = False,
    max_batch_size: int = 32,
    max_batch_wait_ms: float = 5.0,
):
    """
    Start the standalone FastAPI server with ThreadPool or AutoBatch for high throughput.
    
    Args:
        model_name: HuggingFace model name or local path
        host: Host to bind to
        port: Port to bind to
        num_workers: Number of thread pool workers for inference (ThreadPool mode)
        uvicorn_workers: Number of uvicorn worker processes (use 1 for GPU)
        enable_batching: Enable request auto-batching mode
        max_batch_size: Maximum batch size for auto-batching
        max_batch_wait_ms: Maximum time to wait for batch to fill
    """
    import uvicorn
    
    if enable_batching:
        logger.info(f"Starting standalone server with AutoBatch (max_batch={max_batch_size}, max_wait={max_batch_wait_ms}ms)")
    else:
        logger.info(f"Starting standalone server with {num_workers} inference workers")
    
    standalone_app = create_standalone_app(
        model_name=model_name,
        num_workers=num_workers,
        enable_batching=enable_batching,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
    
    # For GPU inference, typically use 1 uvicorn worker to avoid GPU memory issues
    uvicorn.run(
        standalone_app,
        host=host,
        port=port,
        workers=uvicorn_workers,
        log_level="info",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="wtpsplit ONNX API Server with Ray Serve, FastAPI, or BentoML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standalone ThreadPool mode with 8 inference workers
  python serve_onnx_api.py --standalone --num-workers 8

  # Standalone AutoBatch mode (like Triton dynamic batching)
  python serve_onnx_api.py --standalone --auto-batch --max-batch-size 32 --max-batch-wait-ms 5

  # Ray Serve mode with 2 replicas
  python serve_onnx_api.py --num-replicas 2

  # BentoML mode with adaptive batching (RECOMMENDED for production)
  python serve_onnx_api.py --bentoml --max-batch-size 64 --max-batch-wait-ms 10

  # BentoML with high concurrency
  python serve_onnx_api.py --bentoml --max-concurrency 64 --timeout 120

  # Standalone with custom model
  python serve_onnx_api.py --standalone --model segment-any-text/sat-12l --num-workers 4

Modes:
  Ray Serve (default): Scalable deployment with autoscaling replicas
  Standalone ThreadPool: Uses thread pool for concurrent request handling
  Standalone AutoBatch: Collects multiple requests into batches for GPU inference
  BentoML: Production-ready serving with adaptive batching (RECOMMENDED)
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="segment-any-text/sat-3l-sm",
        help="Model name or path (default: segment-any-text/sat-3l-sm)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of Ray Serve replicas (default: 1)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run in standalone mode with ThreadPool (high-throughput, no Ray Serve)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of ThreadPool workers for inference in standalone mode (default: 4)",
    )
    parser.add_argument(
        "--uvicorn-workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1, use 1 for GPU)",
    )
    parser.add_argument(
        "--auto-batch",
        action="store_true",
        help="Enable request auto-batching (like Triton dynamic batching)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for auto-batching (default: 32)",
    )
    parser.add_argument(
        "--max-batch-wait-ms",
        type=float,
        default=5.0,
        help="Maximum time to wait for batch to fill in ms (default: 5.0)",
    )
    parser.add_argument(
        "--bentoml",
        action="store_true",
        help="Run with BentoML (production-ready with adaptive batching)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="Maximum concurrent requests per worker - BentoML mode (default: 32)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds - BentoML mode (default: 300.0)",
    )
    parser.add_argument(
        "--api-workers",
        type=int,
        default=1,
        help="Number of API workers - BentoML mode (default: 1, use 1 for GPU)",
    )
    
    args = parser.parse_args()
    
    if args.bentoml:
        # BentoML mode - production-ready with adaptive batching
        start_bentoml(
            model_name=args.model,
            host=args.host,
            port=args.port,
            max_batch_size=args.max_batch_size,
            max_batch_wait_ms=args.max_batch_wait_ms,
            api_workers=args.api_workers,
            timeout=args.timeout,
            max_concurrency=args.max_concurrency,
        )
    elif args.standalone:
        mode = "AutoBatch" if args.auto_batch else "ThreadPool"
        logger.info("=" * 60)
        logger.info(f" Starting wtpsplit ONNX API (Standalone - {mode})")
        logger.info("=" * 60)
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Host: {args.host}:{args.port}")
        if args.auto_batch:
            logger.info("  Mode: AutoBatch")
            logger.info(f"  Max Batch Size: {args.max_batch_size}")
            logger.info(f"  Max Batch Wait: {args.max_batch_wait_ms} ms")
        else:
            logger.info("  Mode: ThreadPool")
            logger.info(f"  Inference Workers: {args.num_workers}")
        logger.info(f"  Uvicorn Workers: {args.uvicorn_workers}")
        logger.info("=" * 60)
        start_standalone(
            model_name=args.model,
            host=args.host,
            port=args.port,
            num_workers=args.num_workers,
            uvicorn_workers=args.uvicorn_workers,
            enable_batching=args.auto_batch,
            max_batch_size=args.max_batch_size,
            max_batch_wait_ms=args.max_batch_wait_ms,
        )
    else:
        logger.info("=" * 60)
        logger.info(" Starting wtpsplit ONNX API (Ray Serve)")
        logger.info("=" * 60)
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Host: {args.host}:{args.port}")
        logger.info(f"  Replicas: {args.num_replicas}")
        logger.info("=" * 60)
        start_ray_serve(
            model_name=args.model,
            host=args.host,
            port=args.port,
            num_replicas=args.num_replicas,
        )

