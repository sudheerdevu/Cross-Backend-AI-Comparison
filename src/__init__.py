"""
Cross-Backend AI Comparison

A toolkit for comparing AI inference performance across different
hardware backends (ROCm, CUDA, OpenVINO, ONNX Runtime, etc.)
"""

from .backends import (
    Backend,
    BackendConfig,
    ROCmBackend,
    CUDABackend,
    OpenVINOBackend,
    ONNXRuntimeBackend,
    get_available_backends,
)
from .runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    run_benchmark,
    run_comparison,
)
from .analysis import (
    BenchmarkAnalyzer,
    compare_backends,
    generate_comparison_report,
)
from .visualization import (
    plot_latency_comparison,
    plot_throughput_comparison,
    create_comparison_dashboard,
)

__version__ = "1.0.0"
__all__ = [
    # Backends
    "Backend",
    "BackendConfig",
    "ROCmBackend",
    "CUDABackend",
    "OpenVINOBackend",
    "ONNXRuntimeBackend",
    "get_available_backends",
    
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "run_benchmark",
    "run_comparison",
    
    # Analysis
    "BenchmarkAnalyzer",
    "compare_backends",
    "generate_comparison_report",
    
    # Visualization
    "plot_latency_comparison",
    "plot_throughput_comparison",
    "create_comparison_dashboard",
]
