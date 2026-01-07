"""
Benchmark script to compare ONNX GPU runtime vs Triton server inference for SaT models.

Usage:
    # Make sure Triton server is running first:
    # sudo docker run --gpus=all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    #     -v /path/to/triton_models:/models \
    #     nvcr.io/nvidia/tritonserver:24.03-py3 \
    #     tritonserver --model-repository=/models

    # For Ray Serve API, start the server:
    # python serve_onnx_api.py --port 8080

    # For BentoML, start the server:
    # bentoml serve bentoml_service:WtpSplitService --port 8000

    python benchmark_inference.py \
        --model_name segment-any-text/sat-3l-sm \
        --triton_model_name sat_3l_sm \
        --triton_url localhost:8001 \
        --rayserve_url http://localhost:8080 \
        --bentoml_url http://localhost:8000
"""

import argparse
import time
import statistics
from typing import List, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class RayServeClient:
    """Simple HTTP client wrapper for Ray Serve API to match SaT interface."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self._check_health()
    
    def _check_health(self):
        """Check if the Ray Serve API is available."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Ray Serve API not healthy: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ray Serve API at {self.base_url}")
    
    def split(self, text_or_texts):
        """Split text into sentences via HTTP API."""
        if isinstance(text_or_texts, str):
            resp = requests.post(
                f"{self.base_url}/split",
                json={"text": text_or_texts},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["sentences"]
        else:
            # Batch processing
            resp = requests.post(
                f"{self.base_url}/split/batch",
                json={"texts": list(text_or_texts)},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["results"]


class BentoMLClient:
    """Simple HTTP client wrapper for BentoML API to match SaT interface."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self._check_health()
    
    def _check_health(self):
        """Check if the BentoML API is available."""
        try:
            resp = requests.post(f"{self.base_url}/health", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"BentoML API not healthy: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to BentoML API at {self.base_url}")
    
    def split(self, text_or_texts):
        """Split text into sentences via HTTP API."""
        if isinstance(text_or_texts, str):
            # BentoML expects Pydantic model wrapped in field name
            resp = requests.post(
                f"{self.base_url}/split",
                json={"request": {"text": text_or_texts}},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["sentences"]
        else:
            # Batch processing
            resp = requests.post(
                f"{self.base_url}/split",
                json={"request": {"texts": list(text_or_texts)}},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["batch_sentences"]

# Sample texts of varying lengths for benchmarking
SAMPLE_TEXTS = {
    "short": "This is a short test. It has two sentences.",
    "medium": """Machine learning is a subset of artificial intelligence. It enables computers to learn from data. 
Deep learning is a subset of machine learning. Neural networks are the foundation of deep learning. 
These technologies have revolutionized many industries. Natural language processing is one application area.""",
    "long": """Artificial intelligence has transformed the way we interact with technology. From voice assistants 
to recommendation systems, AI is everywhere. Machine learning algorithms can identify patterns in vast amounts 
of data. This capability has led to breakthroughs in healthcare, finance, and transportation. Deep learning 
models can now generate human-like text and images. However, these advances also raise important ethical 
questions. How do we ensure AI systems are fair and unbiased? What happens when AI makes mistakes? These 
are questions society must grapple with. The future of AI holds both tremendous promise and significant 
challenges. Researchers continue to push the boundaries of what's possible. New architectures and training 
methods emerge regularly. The field moves at a breathtaking pace. Keeping up with the latest developments 
requires constant learning. But the potential rewards make it worthwhile. AI could help solve some of 
humanity's greatest challenges. Climate change, disease, and poverty might all be addressed with AI assistance.
The key is to develop these technologies responsibly. We must ensure they benefit everyone, not just a few.""",
}


def warmup(sat_model, text: str, num_warmup: int = 5):
    """Warm up the model to ensure accurate benchmarking."""
    for _ in range(num_warmup):
        sat_model.split(text)


def benchmark_inference(
    sat_model, 
    texts: List[str], 
    num_iterations: int = 100
) -> Tuple[List[float], List[int]]:
    """
    Benchmark inference time for a list of texts.
    
    Returns:
        Tuple of (latencies_ms, char_counts)
    """
    latencies = []
    char_counts = []
    
    for text in texts:
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = sat_model.split(text)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
            char_counts.append(len(text))
    
    return latencies, char_counts


def compute_statistics(latencies: List[float]) -> dict:
    """Compute statistics for latency measurements."""
    return {
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p90_ms": np.percentile(latencies, 90),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
    }


def print_results(name: str, stats: dict, total_chars: int, total_time_ms: float):
    """Print benchmark results in a formatted table."""
    throughput_chars_per_sec = (total_chars / total_time_ms) * 1000
    
    print(f"\n{'=' * 60}")
    print(f" {name} Results")
    print(f"{'=' * 60}")
    print(f"  Mean latency:      {stats['mean_ms']:>10.2f} ms")
    print(f"  Median latency:    {stats['median_ms']:>10.2f} ms")
    print(f"  Std deviation:     {stats['std_ms']:>10.2f} ms")
    print(f"  Min latency:       {stats['min_ms']:>10.2f} ms")
    print(f"  Max latency:       {stats['max_ms']:>10.2f} ms")
    print(f"  P50 latency:       {stats['p50_ms']:>10.2f} ms")
    print(f"  P90 latency:       {stats['p90_ms']:>10.2f} ms")
    print(f"  P95 latency:       {stats['p95_ms']:>10.2f} ms")
    print(f"  P99 latency:       {stats['p99_ms']:>10.2f} ms")
    print(f"  Throughput:        {throughput_chars_per_sec:>10.0f} chars/sec")
    print(f"{'=' * 60}")


def run_benchmark(
    model_name: str,
    triton_url: str = None,
    triton_model_name: str = None,
    rayserve_url: str = None,
    bentoml_url: str = None,
    num_iterations: int = 100,
    num_warmup: int = 10,
    text_sizes: List[str] = None,
    skip_onnx: bool = False,
):
    """Run the complete benchmark comparing ONNX, Triton, Ray Serve, and BentoML inference."""
    from wtpsplit import SaT
    
    if text_sizes is None:
        text_sizes = ["short", "medium", "long"]
    
    texts = [SAMPLE_TEXTS[size] for size in text_sizes]
    
    results = {}
    
    # Benchmark ONNX GPU Runtime
    if not skip_onnx:
        print("\n" + "=" * 60)
        print(" Benchmarking ONNX GPU Runtime")
        print("=" * 60)
        
        try:
            sat_onnx = SaT(
                model_name,
                ort_providers=["CUDAExecutionProvider"],
            )
            
            print("Warming up ONNX model...")
            warmup(sat_onnx, texts[0], num_warmup)
            
            print(f"Running {num_iterations} iterations per text...")
            onnx_latencies, onnx_chars = benchmark_inference(sat_onnx, texts, num_iterations)
            
            onnx_stats = compute_statistics(onnx_latencies)
            total_chars = sum(onnx_chars)
            total_time = sum(onnx_latencies)
            
            print_results("ONNX GPU Runtime", onnx_stats, total_chars, total_time)
            results["onnx"] = {
                "stats": onnx_stats,
                "total_chars": total_chars,
                "total_time_ms": total_time,
            }
            
            # Clean up
            del sat_onnx
            
        except Exception as e:
            print(f"ONNX benchmark failed: {e}")
            results["onnx"] = None
    
    # Benchmark Triton Server
    if triton_url and triton_model_name:
        print("\n" + "=" * 60)
        print(" Benchmarking Triton Server")
        print("=" * 60)
        
        try:
            sat_triton = SaT(
                model_name,
                triton_url=triton_url,
                triton_model_name=triton_model_name,
            )
            
            print("Warming up Triton model...")
            warmup(sat_triton, texts[0], num_warmup)
            
            print(f"Running {num_iterations} iterations per text...")
            triton_latencies, triton_chars = benchmark_inference(sat_triton, texts, num_iterations)
            
            triton_stats = compute_statistics(triton_latencies)
            total_chars = sum(triton_chars)
            total_time = sum(triton_latencies)
            
            print_results("Triton Server", triton_stats, total_chars, total_time)
            results["triton"] = {
                "stats": triton_stats,
                "total_chars": total_chars,
                "total_time_ms": total_time,
            }
            
            # Clean up
            del sat_triton
            
        except Exception as e:
            print(f"Triton benchmark failed: {e}")
            results["triton"] = None
    
    # Benchmark Ray Serve API
    if rayserve_url:
        print("\n" + "=" * 60)
        print(" Benchmarking Ray Serve ONNX API")
        print("=" * 60)
        
        if not REQUESTS_AVAILABLE:
            print("Ray Serve benchmark skipped: 'requests' library not installed")
            results["rayserve"] = None
        else:
            try:
                rayserve_client = RayServeClient(rayserve_url)
                
                print("Warming up Ray Serve API...")
                warmup(rayserve_client, texts[0], num_warmup)
                
                print(f"Running {num_iterations} iterations per text...")
                rayserve_latencies, rayserve_chars = benchmark_inference(rayserve_client, texts, num_iterations)
                
                rayserve_stats = compute_statistics(rayserve_latencies)
                total_chars = sum(rayserve_chars)
                total_time = sum(rayserve_latencies)
                
                print_results("Ray Serve ONNX API", rayserve_stats, total_chars, total_time)
                results["rayserve"] = {
                    "stats": rayserve_stats,
                    "total_chars": total_chars,
                    "total_time_ms": total_time,
                }
                
            except Exception as e:
                print(f"Ray Serve benchmark failed: {e}")
                results["rayserve"] = None
    
    # Benchmark BentoML API
    if bentoml_url:
        print("\n" + "=" * 60)
        print(" Benchmarking BentoML ONNX API")
        print("=" * 60)
        
        if not REQUESTS_AVAILABLE:
            print("BentoML benchmark skipped: 'requests' library not installed")
            results["bentoml"] = None
        else:
            try:
                bentoml_client = BentoMLClient(bentoml_url)
                
                print("Warming up BentoML API...")
                warmup(bentoml_client, texts[0], num_warmup)
                
                print(f"Running {num_iterations} iterations per text...")
                bentoml_latencies, bentoml_chars = benchmark_inference(bentoml_client, texts, num_iterations)
                
                bentoml_stats = compute_statistics(bentoml_latencies)
                total_chars = sum(bentoml_chars)
                total_time = sum(bentoml_latencies)
                
                print_results("BentoML ONNX API", bentoml_stats, total_chars, total_time)
                results["bentoml"] = {
                    "stats": bentoml_stats,
                    "total_chars": total_chars,
                    "total_time_ms": total_time,
                }
                
            except Exception as e:
                print(f"BentoML benchmark failed: {e}")
                results["bentoml"] = None
    
    # Compare results
    available_results = {k: v for k, v in results.items() if v is not None}
    if len(available_results) >= 2:
        print("\n" + "=" * 60)
        print(" Comparison Summary")
        print("=" * 60)
        
        # Print all latencies
        for name, data in available_results.items():
            mean_lat = data["stats"]["mean_ms"]
            throughput = (data["total_chars"] / data["total_time_ms"]) * 1000
            print(f"  {name:>12} mean latency: {mean_lat:>10.2f} ms | throughput: {throughput:>10.0f} chars/sec")
        
        # Find the fastest
        fastest = min(available_results.items(), key=lambda x: x[1]["stats"]["mean_ms"])
        print(f"\n  → Fastest: {fastest[0]} ({fastest[1]['stats']['mean_ms']:.2f} ms)")
        
        print("=" * 60)
    
    return results


def run_batch_benchmark(
    model_name: str,
    triton_url: str = None,
    triton_model_name: str = None,
    rayserve_url: str = None,
    bentoml_url: str = None,
    batch_sizes: List[int] = None,
    num_iterations: int = 50,
    skip_onnx: bool = False,
):
    """Benchmark with different batch sizes (multiple texts at once)."""
    from wtpsplit import SaT
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32]
    
    base_text = SAMPLE_TEXTS["medium"]
    
    print("\n" + "=" * 60)
    print(" Batch Size Benchmark")
    print("=" * 60)
    
    results = {"onnx": {}, "triton": {}, "rayserve": {}, "bentoml": {}}
    
    # ONNX benchmark
    if not skip_onnx:
        try:
            sat_onnx = SaT(model_name, ort_providers=["CUDAExecutionProvider"])
            
            print("\nONNX GPU Runtime - Batch benchmarks:")
            for batch_size in batch_sizes:
                batch_texts = [base_text] * batch_size
                total_chars = batch_size * len(base_text)
                
                # Warmup
                for _ in range(5):
                    # Force evaluation by converting to list
                    _ = list(sat_onnx.split(batch_texts))
                
                latencies = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    # Force evaluation by converting to list
                    _ = list(sat_onnx.split(batch_texts))
                    end = time.perf_counter()
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)
                
                mean_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
                throughput = (total_chars / mean_latency) * 1000 if mean_latency > 0 else 0
                
                print(f"  Batch size {batch_size:>3}: {mean_latency:>8.2f} ms ± {std_latency:>6.2f}, {throughput:>10.0f} chars/sec")
                results["onnx"][batch_size] = {"mean_ms": mean_latency, "std_ms": std_latency, "throughput": throughput}
            
            del sat_onnx
        except Exception as e:
            print(f"ONNX batch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Triton benchmark
    if triton_url and triton_model_name:
        try:
            sat_triton = SaT(model_name, triton_url=triton_url, triton_model_name=triton_model_name)
            
            print("\nTriton Server - Batch benchmarks:")
            for batch_size in batch_sizes:
                batch_texts = [base_text] * batch_size
                total_chars = batch_size * len(base_text)
                
                # Warmup
                for _ in range(5):
                    # Force evaluation by converting to list
                    _ = list(sat_triton.split(batch_texts))
                
                latencies = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    # Force evaluation by converting to list
                    _ = list(sat_triton.split(batch_texts))
                    end = time.perf_counter()
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)
                
                mean_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
                throughput = (total_chars / mean_latency) * 1000 if mean_latency > 0 else 0
                
                print(f"  Batch size {batch_size:>3}: {mean_latency:>8.2f} ms ± {std_latency:>6.2f}, {throughput:>10.0f} chars/sec")
                results["triton"][batch_size] = {"mean_ms": mean_latency, "std_ms": std_latency, "throughput": throughput}
            
            del sat_triton
        except Exception as e:
            print(f"Triton batch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Ray Serve benchmark
    if rayserve_url and REQUESTS_AVAILABLE:
        try:
            rayserve_client = RayServeClient(rayserve_url)
            
            print("\nRay Serve ONNX API - Batch benchmarks:")
            for batch_size in batch_sizes:
                batch_texts = [base_text] * batch_size
                total_chars = batch_size * len(base_text)
                
                # Warmup
                for _ in range(5):
                    _ = rayserve_client.split(batch_texts)
                
                latencies = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = rayserve_client.split(batch_texts)
                    end = time.perf_counter()
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)
                
                mean_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
                throughput = (total_chars / mean_latency) * 1000 if mean_latency > 0 else 0
                
                print(f"  Batch size {batch_size:>3}: {mean_latency:>8.2f} ms ± {std_latency:>6.2f}, {throughput:>10.0f} chars/sec")
                results["rayserve"][batch_size] = {"mean_ms": mean_latency, "std_ms": std_latency, "throughput": throughput}
            
        except Exception as e:
            print(f"Ray Serve batch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # BentoML benchmark
    if bentoml_url and REQUESTS_AVAILABLE:
        try:
            bentoml_client = BentoMLClient(bentoml_url)
            
            print("\nBentoML ONNX API - Batch benchmarks:")
            for batch_size in batch_sizes:
                batch_texts = [base_text] * batch_size
                total_chars = batch_size * len(base_text)
                
                # Warmup
                for _ in range(5):
                    _ = bentoml_client.split(batch_texts)
                
                latencies = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = bentoml_client.split(batch_texts)
                    end = time.perf_counter()
                    latency_ms = (end - start) * 1000
                    latencies.append(latency_ms)
                
                mean_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
                throughput = (total_chars / mean_latency) * 1000 if mean_latency > 0 else 0
                
                print(f"  Batch size {batch_size:>3}: {mean_latency:>8.2f} ms ± {std_latency:>6.2f}, {throughput:>10.0f} chars/sec")
                results["bentoml"][batch_size] = {"mean_ms": mean_latency, "std_ms": std_latency, "throughput": throughput}
            
        except Exception as e:
            print(f"BentoML batch benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def run_concurrent_benchmark(
    model_name: str,
    triton_url: str = None,
    triton_model_name: str = None,
    rayserve_url: str = None,
    bentoml_url: str = None,
    num_clients: List[int] = None,
    requests_per_client: int = 20,
    skip_onnx: bool = False,
):
    """
    Benchmark with multiple concurrent clients to test throughput under load.
    This is where Triton, Ray Serve, and BentoML should shine due to their request handling.
    """
    from wtpsplit import SaT
    
    if num_clients is None:
        num_clients = [1, 2, 4, 8, 16, 32]
    
    base_text = SAMPLE_TEXTS["medium"]
    total_chars_per_request = len(base_text)
    
    print("\n" + "=" * 70)
    print(" Concurrent Clients Benchmark (simulates production load)")
    print("=" * 70)
    print(f"  Requests per client: {requests_per_client}")
    print(f"  Text length: {total_chars_per_request} chars")
    print("=" * 70)
    
    results = {"onnx": {}, "triton": {}, "rayserve": {}, "bentoml": {}}
    
    def client_worker(sat_model, text: str, num_requests: int) -> List[float]:
        """Worker function for a single client."""
        latencies = []
        for _ in range(num_requests):
            start = time.perf_counter()
            _ = sat_model.split(text)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        return latencies
    
    # ONNX benchmark
    if not skip_onnx:
        try:
            sat_onnx = SaT(model_name, ort_providers=["CUDAExecutionProvider"])
            
            # Warmup
            for _ in range(10):
                _ = sat_onnx.split(base_text)
            
            print("\nONNX GPU Runtime - Concurrent clients:")
            print(f"  {'Clients':>8} | {'Total Time':>12} | {'Throughput':>15} | {'Avg Latency':>12} | {'P99 Latency':>12}")
            print("  " + "-" * 68)
            
            for n_clients in num_clients:
                all_latencies = []
                
                # Run concurrent clients
                start_time = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_clients) as executor:
                    futures = [
                        executor.submit(client_worker, sat_onnx, base_text, requests_per_client)
                        for _ in range(n_clients)
                    ]
                    for future in as_completed(futures):
                        all_latencies.extend(future.result())
                end_time = time.perf_counter()
                
                total_time_sec = end_time - start_time
                total_requests = n_clients * requests_per_client
                total_chars = total_requests * total_chars_per_request
                throughput = total_chars / total_time_sec
                avg_latency = statistics.mean(all_latencies)
                p99_latency = np.percentile(all_latencies, 99)
                
                print(f"  {n_clients:>8} | {total_time_sec:>10.2f} s | {throughput:>12.0f} c/s | {avg_latency:>10.2f} ms | {p99_latency:>10.2f} ms")
                results["onnx"][n_clients] = {
                    "total_time_sec": total_time_sec,
                    "throughput_chars_sec": throughput,
                    "avg_latency_ms": avg_latency,
                    "p99_latency_ms": p99_latency,
                }
            
            del sat_onnx
        except Exception as e:
            print(f"ONNX concurrent benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Triton benchmark
    if triton_url and triton_model_name:
        try:
            sat_triton = SaT(model_name, triton_url=triton_url, triton_model_name=triton_model_name)
            
            # Warmup
            for _ in range(10):
                _ = sat_triton.split(base_text)
            
            print("\nTriton Server - Concurrent clients:")
            print(f"  {'Clients':>8} | {'Total Time':>12} | {'Throughput':>15} | {'Avg Latency':>12} | {'P99 Latency':>12}")
            print("  " + "-" * 68)
            
            for n_clients in num_clients:
                all_latencies = []
                
                # Run concurrent clients
                start_time = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_clients) as executor:
                    futures = [
                        executor.submit(client_worker, sat_triton, base_text, requests_per_client)
                        for _ in range(n_clients)
                    ]
                    for future in as_completed(futures):
                        all_latencies.extend(future.result())
                end_time = time.perf_counter()
                
                total_time_sec = end_time - start_time
                total_requests = n_clients * requests_per_client
                total_chars = total_requests * total_chars_per_request
                throughput = total_chars / total_time_sec
                avg_latency = statistics.mean(all_latencies)
                p99_latency = np.percentile(all_latencies, 99)
                
                print(f"  {n_clients:>8} | {total_time_sec:>10.2f} s | {throughput:>12.0f} c/s | {avg_latency:>10.2f} ms | {p99_latency:>10.2f} ms")
                results["triton"][n_clients] = {
                    "total_time_sec": total_time_sec,
                    "throughput_chars_sec": throughput,
                    "avg_latency_ms": avg_latency,
                    "p99_latency_ms": p99_latency,
                }
            
            del sat_triton
        except Exception as e:
            print(f"Triton concurrent benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Ray Serve benchmark
    if rayserve_url and REQUESTS_AVAILABLE:
        try:
            rayserve_client = RayServeClient(rayserve_url)
            
            # Warmup
            for _ in range(10):
                _ = rayserve_client.split(base_text)
            
            print("\nRay Serve ONNX API - Concurrent clients:")
            print(f"  {'Clients':>8} | {'Total Time':>12} | {'Throughput':>15} | {'Avg Latency':>12} | {'P99 Latency':>12}")
            print("  " + "-" * 68)
            
            for n_clients in num_clients:
                all_latencies = []
                
                # Run concurrent clients
                start_time = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_clients) as executor:
                    futures = [
                        executor.submit(client_worker, rayserve_client, base_text, requests_per_client)
                        for _ in range(n_clients)
                    ]
                    for future in as_completed(futures):
                        all_latencies.extend(future.result())
                end_time = time.perf_counter()
                
                total_time_sec = end_time - start_time
                total_requests = n_clients * requests_per_client
                total_chars = total_requests * total_chars_per_request
                throughput = total_chars / total_time_sec
                avg_latency = statistics.mean(all_latencies)
                p99_latency = np.percentile(all_latencies, 99)
                
                print(f"  {n_clients:>8} | {total_time_sec:>10.2f} s | {throughput:>12.0f} c/s | {avg_latency:>10.2f} ms | {p99_latency:>10.2f} ms")
                results["rayserve"][n_clients] = {
                    "total_time_sec": total_time_sec,
                    "throughput_chars_sec": throughput,
                    "avg_latency_ms": avg_latency,
                    "p99_latency_ms": p99_latency,
                }
            
        except Exception as e:
            print(f"Ray Serve concurrent benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # BentoML benchmark
    if bentoml_url and REQUESTS_AVAILABLE:
        try:
            bentoml_client = BentoMLClient(bentoml_url)
            
            # Warmup
            for _ in range(10):
                _ = bentoml_client.split(base_text)
            
            print("\nBentoML ONNX API - Concurrent clients:")
            print(f"  {'Clients':>8} | {'Total Time':>12} | {'Throughput':>15} | {'Avg Latency':>12} | {'P99 Latency':>12}")
            print("  " + "-" * 68)
            
            for n_clients in num_clients:
                all_latencies = []
                
                # Run concurrent clients
                start_time = time.perf_counter()
                with ThreadPoolExecutor(max_workers=n_clients) as executor:
                    futures = [
                        executor.submit(client_worker, bentoml_client, base_text, requests_per_client)
                        for _ in range(n_clients)
                    ]
                    for future in as_completed(futures):
                        all_latencies.extend(future.result())
                end_time = time.perf_counter()
                
                total_time_sec = end_time - start_time
                total_requests = n_clients * requests_per_client
                total_chars = total_requests * total_chars_per_request
                throughput = total_chars / total_time_sec
                avg_latency = statistics.mean(all_latencies)
                p99_latency = np.percentile(all_latencies, 99)
                
                print(f"  {n_clients:>8} | {total_time_sec:>10.2f} s | {throughput:>12.0f} c/s | {avg_latency:>10.2f} ms | {p99_latency:>10.2f} ms")
                results["bentoml"][n_clients] = {
                    "total_time_sec": total_time_sec,
                    "throughput_chars_sec": throughput,
                    "avg_latency_ms": avg_latency,
                    "p99_latency_ms": p99_latency,
                }
            
        except Exception as e:
            print(f"BentoML concurrent benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison
    available = {k: v for k, v in results.items() if v}
    if len(available) >= 2:
        print("\n" + "=" * 70)
        print(" Concurrent Benchmark Comparison")
        print("=" * 70)
        
        # Build header dynamically
        header = f"  {'Clients':>8}"
        for name in available.keys():
            header += f" | {name:>15}"
        print(header)
        print("  " + "-" * (10 + 18 * len(available)))
        
        for n_clients in num_clients:
            row = f"  {n_clients:>8}"
            for name, data in available.items():
                if n_clients in data:
                    tp = data[n_clients]["throughput_chars_sec"]
                    row += f" | {tp:>12.0f} c/s"
                else:
                    row += f" | {'N/A':>15}"
            print(row)
        
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX vs Triton vs Ray Serve inference for SaT models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="segment-any-text/sat-3l-sm",
        help="Model name or path",
    )
    parser.add_argument(
        "--triton_url",
        type=str,
        default="localhost:8001",
        help="Triton server URL (gRPC)",
    )
    parser.add_argument(
        "--triton_model_name",
        type=str,
        default="sat_3l_sm",
        help="Triton model name",
    )
    parser.add_argument(
        "--rayserve_url",
        type=str,
        default=None,
        help="Ray Serve API URL (e.g., http://localhost:8080)",
    )
    parser.add_argument(
        "--bentoml_url",
        type=str,
        default=None,
        help="BentoML API URL (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--skip_onnx",
        action="store_true",
        help="Skip ONNX benchmark",
    )
    parser.add_argument(
        "--skip_triton",
        action="store_true",
        help="Skip Triton benchmark",
    )
    parser.add_argument(
        "--skip_rayserve",
        action="store_true",
        help="Skip Ray Serve benchmark",
    )
    parser.add_argument(
        "--skip_bentoml",
        action="store_true",
        help="Skip BentoML benchmark",
    )
    parser.add_argument(
        "--batch_benchmark",
        action="store_true",
        help="Run batch size benchmark",
    )
    parser.add_argument(
        "--concurrent_benchmark",
        action="store_true",
        help="Run concurrent clients benchmark (simulates production load)",
    )
    parser.add_argument(
        "--requests_per_client",
        type=int,
        default=20,
        help="Number of requests per client in concurrent benchmark",
    )
    
    args = parser.parse_args()
    
    # Determine which backends to benchmark
    triton_url = None if args.skip_triton else args.triton_url
    triton_model_name = None if args.skip_triton else args.triton_model_name
    rayserve_url = None if args.skip_rayserve else args.rayserve_url
    bentoml_url = None if args.skip_bentoml else args.bentoml_url
    
    print("\n" + "=" * 60)
    print(" SaT Inference Benchmark: ONNX vs Triton vs Ray Serve vs BentoML")
    print("=" * 60)
    print(f"  Model: {args.model_name}")
    print("  Backends:")
    if not args.skip_onnx:
        print("    - ONNX GPU Runtime (local)")
    if triton_url:
        print(f"    - Triton Server: {triton_url} (model: {triton_model_name})")
    if rayserve_url:
        print(f"    - Ray Serve API: {rayserve_url}")
    if bentoml_url:
        print(f"    - BentoML API: {bentoml_url}")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Warmup: {args.num_warmup}")
    print("=" * 60)
    
    # Run main benchmark
    run_benchmark(
        model_name=args.model_name,
        triton_url=triton_url,
        triton_model_name=triton_model_name,
        rayserve_url=rayserve_url,
        bentoml_url=bentoml_url,
        num_iterations=args.num_iterations,
        num_warmup=args.num_warmup,
        skip_onnx=args.skip_onnx,
    )
    
    # Run batch benchmark if requested
    if args.batch_benchmark:
        run_batch_benchmark(
            model_name=args.model_name,
            triton_url=triton_url,
            triton_model_name=triton_model_name,
            rayserve_url=rayserve_url,
            bentoml_url=bentoml_url,
            skip_onnx=args.skip_onnx,
        )
    
    # Run concurrent benchmark if requested
    if args.concurrent_benchmark:
        run_concurrent_benchmark(
            model_name=args.model_name,
            triton_url=triton_url,
            triton_model_name=triton_model_name,
            rayserve_url=rayserve_url,
            bentoml_url=bentoml_url,
            requests_per_client=args.requests_per_client,
            skip_onnx=args.skip_onnx,
        )
    
    print("\nBenchmark complete!")
