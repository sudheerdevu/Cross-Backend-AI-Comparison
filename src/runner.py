#!/usr/bin/env python3
"""
Cross-Backend AI Comparison - Test Runner

Systematic comparison framework for AI inference across multiple backends.
"""

import onnxruntime as ort
import numpy as np
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import platform


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    backend: str
    model: str
    batch_size: int
    latency_ms: float
    latency_p50: float
    latency_p90: float
    latency_p99: float
    throughput: float  # inferences/sec
    memory_mb: float
    first_inference_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Abstract base class for inference backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available on this system."""
        pass
    
    @abstractmethod
    def create_session(self, model_path: str) -> Any:
        """Create inference session."""
        pass
    
    @abstractmethod
    def run_inference(self, session: Any, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference."""
        pass


class CPUBackend(Backend):
    """CPU backend using ONNX Runtime."""
    
    @property
    def name(self) -> str:
        return "cpu"
    
    def is_available(self) -> bool:
        return True
    
    def create_session(self, model_path: str) -> ort.InferenceSession:
        providers = ['CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0  # Use all cores
        return ort.InferenceSession(model_path, opts, providers=providers)
    
    def run_inference(self, session: ort.InferenceSession, 
                      inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return session.run(None, inputs)


class CUDABackend(Backend):
    """NVIDIA CUDA backend."""
    
    @property
    def name(self) -> str:
        return "cuda"
    
    def is_available(self) -> bool:
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    
    def create_session(self, model_path: str) -> ort.InferenceSession:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ort.InferenceSession(model_path, providers=providers)
    
    def run_inference(self, session: ort.InferenceSession,
                      inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return session.run(None, inputs)


class ROCmBackend(Backend):
    """AMD ROCm backend using MIGraphX or ROCm EP."""
    
    @property
    def name(self) -> str:
        return "rocm"
    
    def is_available(self) -> bool:
        providers = ort.get_available_providers()
        return 'ROCMExecutionProvider' in providers or 'MIGraphXExecutionProvider' in providers
    
    def create_session(self, model_path: str) -> ort.InferenceSession:
        providers = []
        available = ort.get_available_providers()
        
        if 'MIGraphXExecutionProvider' in available:
            providers.append('MIGraphXExecutionProvider')
        if 'ROCMExecutionProvider' in available:
            providers.append('ROCMExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        return ort.InferenceSession(model_path, providers=providers)
    
    def run_inference(self, session: ort.InferenceSession,
                      inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return session.run(None, inputs)


class DirectMLBackend(Backend):
    """DirectML backend for Windows."""
    
    @property
    def name(self) -> str:
        return "directml"
    
    def is_available(self) -> bool:
        if platform.system() != 'Windows':
            return False
        return 'DmlExecutionProvider' in ort.get_available_providers()
    
    def create_session(self, model_path: str) -> ort.InferenceSession:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        return ort.InferenceSession(model_path, providers=providers)
    
    def run_inference(self, session: ort.InferenceSession,
                      inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return session.run(None, inputs)


class ComparisonRunner:
    """Run comparison benchmarks across backends."""
    
    BACKENDS = {
        'cpu': CPUBackend,
        'cuda': CUDABackend,
        'rocm': ROCmBackend,
        'directml': DirectMLBackend,
    }
    
    def __init__(self, 
                 model_path: str,
                 backends: List[str] = None,
                 batch_sizes: List[int] = None,
                 warmup_iterations: int = 10,
                 benchmark_iterations: int = 100):
        self.model_path = model_path
        self.batch_sizes = batch_sizes or [1, 4, 16]
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        
        # Initialize requested backends
        self.backends: List[Backend] = []
        backend_names = backends or ['cpu']
        
        for name in backend_names:
            if name in self.BACKENDS:
                backend = self.BACKENDS[name]()
                if backend.is_available():
                    self.backends.append(backend)
                    print(f"Backend '{name}' available")
                else:
                    print(f"Backend '{name}' not available on this system")
    
    def generate_input(self, session: ort.InferenceSession, 
                       batch_size: int) -> Dict[str, np.ndarray]:
        """Generate random input for the model."""
        inputs = {}
        for inp in session.get_inputs():
            shape = list(inp.shape)
            # Replace dynamic dimensions
            shape[0] = batch_size
            for i, dim in enumerate(shape):
                if isinstance(dim, str) or dim is None:
                    shape[i] = 128  # Default for dynamic dims
            
            dtype = np.float32
            if inp.type == 'tensor(int64)':
                dtype = np.int64
            elif inp.type == 'tensor(int32)':
                dtype = np.int32
            
            inputs[inp.name] = np.random.randn(*shape).astype(dtype)
        
        return inputs
    
    def benchmark_backend(self, backend: Backend, 
                          batch_size: int) -> BenchmarkResult:
        """Benchmark a single backend."""
        # Create session
        session = backend.create_session(self.model_path)
        inputs = self.generate_input(session, batch_size)
        
        # First inference (cold start)
        start = time.perf_counter()
        backend.run_inference(session, inputs)
        first_inference_ms = (time.perf_counter() - start) * 1000
        
        # Warmup
        for _ in range(self.warmup_iterations):
            backend.run_inference(session, inputs)
        
        # Benchmark
        latencies = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            backend.run_inference(session, inputs)
            latencies.append((time.perf_counter() - start) * 1000)
        
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        
        return BenchmarkResult(
            backend=backend.name,
            model=Path(self.model_path).stem,
            batch_size=batch_size,
            latency_ms=avg_latency,
            latency_p50=float(np.percentile(latencies, 50)),
            latency_p90=float(np.percentile(latencies, 90)),
            latency_p99=float(np.percentile(latencies, 99)),
            throughput=batch_size / (avg_latency / 1000),
            memory_mb=0,  # Would need backend-specific measurement
            first_inference_ms=first_inference_ms,
            metadata={
                'warmup_iterations': self.warmup_iterations,
                'benchmark_iterations': self.benchmark_iterations
            }
        )
    
    def run(self) -> 'ComparisonResults':
        """Run full comparison across all backends and batch sizes."""
        results = []
        
        for backend in self.backends:
            for batch_size in self.batch_sizes:
                print(f"Benchmarking {backend.name} with batch_size={batch_size}...")
                try:
                    result = self.benchmark_backend(backend, batch_size)
                    results.append(result)
                    print(f"  Latency: {result.latency_ms:.2f}ms, "
                          f"Throughput: {result.throughput:.1f} inf/s")
                except Exception as e:
                    print(f"  Error: {e}")
        
        return ComparisonResults(results)


class ComparisonResults:
    """Container and analysis for comparison results."""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([
            {
                'backend': r.backend,
                'model': r.model,
                'batch_size': r.batch_size,
                'latency_ms': r.latency_ms,
                'latency_p50': r.latency_p50,
                'latency_p99': r.latency_p99,
                'throughput': r.throughput,
                'first_inference_ms': r.first_inference_ms
            }
            for r in self.results
        ])
    
    def to_json(self, path: str) -> None:
        """Save results to JSON."""
        data = [
            {
                'backend': r.backend,
                'model': r.model,
                'batch_size': r.batch_size,
                'latency_ms': r.latency_ms,
                'latency_p50': r.latency_p50,
                'latency_p90': r.latency_p90,
                'latency_p99': r.latency_p99,
                'throughput': r.throughput,
                'first_inference_ms': r.first_inference_ms
            }
            for r in self.results
        ]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_report(self, output_path: str = "report.md") -> str:
        """Generate markdown report."""
        lines = [
            "# Cross-Backend AI Comparison Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Results Summary",
            "",
            "| Backend | Batch Size | Latency (ms) | P99 (ms) | Throughput |",
            "|---------|------------|--------------|----------|------------|",
        ]
        
        for r in self.results:
            lines.append(
                f"| {r.backend} | {r.batch_size} | {r.latency_ms:.2f} | "
                f"{r.latency_p99:.2f} | {r.throughput:.1f} |"
            )
        
        lines.extend(["", "## Backend Comparison", ""])
        
        # Find best backend for each batch size
        by_batch: Dict[int, List[BenchmarkResult]] = {}
        for r in self.results:
            if r.batch_size not in by_batch:
                by_batch[r.batch_size] = []
            by_batch[r.batch_size].append(r)
        
        for batch, batch_results in sorted(by_batch.items()):
            best = min(batch_results, key=lambda x: x.latency_ms)
            lines.append(f"### Batch Size {batch}")
            lines.append(f"Best: **{best.backend}** ({best.latency_ms:.2f}ms)")
            lines.append("")
        
        report = '\n'.join(lines)
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-backend comparison')
    parser.add_argument('model', help='Path to ONNX model')
    parser.add_argument('--backends', nargs='+', 
                       default=['cpu', 'cuda', 'rocm', 'directml'],
                       help='Backends to test')
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                       default=[1, 4, 16], help='Batch sizes to test')
    parser.add_argument('--output', default='comparison_results.json',
                       help='Output file')
    
    args = parser.parse_args()
    
    runner = ComparisonRunner(
        args.model,
        backends=args.backends,
        batch_sizes=args.batch_sizes
    )
    
    results = runner.run()
    results.to_json(args.output)
    results.generate_report('comparison_report.md')
    
    print(f"\nResults saved to {args.output}")
