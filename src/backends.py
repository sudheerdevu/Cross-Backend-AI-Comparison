"""
Cross-Backend AI Comparison - Backend Implementations

This module provides unified interfaces to different inference backends.
"""

import abc
import time
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class InferenceResult:
    """Result from a single inference run"""
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    memory_used_mb: float
    backend: str


class BaseBackend(abc.ABC):
    """Abstract base class for inference backends"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_initialized = False
    
    @abc.abstractmethod
    def initialize(self, model_path: str, **kwargs) -> None:
        """Initialize the backend with a model"""
        pass
    
    @abc.abstractmethod
    def inference(self, inputs: Dict[str, np.ndarray]) -> InferenceResult:
        """Run inference on the given inputs"""
        pass
    
    @abc.abstractmethod
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass


class ONNXRuntimeCPUBackend(BaseBackend):
    """ONNX Runtime CPU backend"""
    
    def __init__(self):
        super().__init__("onnxruntime-cpu")
        self.session = None
    
    def initialize(self, model_path: str, **kwargs) -> None:
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = kwargs.get('num_threads', 4)
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_initialized = True
    
    def inference(self, inputs: Dict[str, np.ndarray]) -> InferenceResult:
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, inputs)
        latency = (time.perf_counter() - start) * 1000
        
        return InferenceResult(
            outputs=dict(zip(self.output_names, outputs)),
            latency_ms=latency,
            memory_used_mb=self.get_memory_usage(),
            backend=self.name
        )
    
    def get_memory_usage(self) -> float:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def cleanup(self) -> None:
        self.session = None


class ONNXRuntimeCUDABackend(BaseBackend):
    """ONNX Runtime CUDA backend"""
    
    def __init__(self):
        super().__init__("onnxruntime-cuda")
        self.session = None
    
    def initialize(self, model_path: str, **kwargs) -> None:
        import onnxruntime as ort
        
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': kwargs.get('device_id', 0),
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_initialized = True
    
    def inference(self, inputs: Dict[str, np.ndarray]) -> InferenceResult:
        # Warmup for accurate timing
        import torch
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, inputs)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        
        return InferenceResult(
            outputs=dict(zip(self.output_names, outputs)),
            latency_ms=latency,
            memory_used_mb=self.get_memory_usage(),
            backend=self.name
        )
    
    def get_memory_usage(self) -> float:
        try:
            import torch
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            return 0.0
    
    def cleanup(self) -> None:
        self.session = None
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass


class ONNXRuntimeROCmBackend(BaseBackend):
    """ONNX Runtime ROCm/MIGraphX backend"""
    
    def __init__(self):
        super().__init__("onnxruntime-rocm")
        self.session = None
    
    def initialize(self, model_path: str, **kwargs) -> None:
        import onnxruntime as ort
        
        providers = [
            ('ROCMExecutionProvider', {
                'device_id': kwargs.get('device_id', 0),
            }),
            'CPUExecutionProvider'
        ]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_initialized = True
    
    def inference(self, inputs: Dict[str, np.ndarray]) -> InferenceResult:
        import torch
        torch.cuda.synchronize()  # ROCm uses same API
        
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, inputs)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        
        return InferenceResult(
            outputs=dict(zip(self.output_names, outputs)),
            latency_ms=latency,
            memory_used_mb=self.get_memory_usage(),
            backend=self.name
        )
    
    def get_memory_usage(self) -> float:
        try:
            import torch
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            return 0.0
    
    def cleanup(self) -> None:
        self.session = None


class DirectMLBackend(BaseBackend):
    """DirectML backend for Windows"""
    
    def __init__(self):
        super().__init__("directml")
        self.session = None
    
    def initialize(self, model_path: str, **kwargs) -> None:
        import onnxruntime as ort
        
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_initialized = True
    
    def inference(self, inputs: Dict[str, np.ndarray]) -> InferenceResult:
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, inputs)
        latency = (time.perf_counter() - start) * 1000
        
        return InferenceResult(
            outputs=dict(zip(self.output_names, outputs)),
            latency_ms=latency,
            memory_used_mb=self.get_memory_usage(),
            backend=self.name
        )
    
    def get_memory_usage(self) -> float:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def cleanup(self) -> None:
        self.session = None


class TensorRTBackend(BaseBackend):
    """TensorRT backend for NVIDIA GPUs"""
    
    def __init__(self):
        super().__init__("tensorrt")
        self.session = None
    
    def initialize(self, model_path: str, **kwargs) -> None:
        import onnxruntime as ort
        
        providers = [
            ('TensorrtExecutionProvider', {
                'device_id': kwargs.get('device_id', 0),
                'trt_fp16_enable': kwargs.get('fp16', True),
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_initialized = True
    
    def inference(self, inputs: Dict[str, np.ndarray]) -> InferenceResult:
        import torch
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, inputs)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        
        return InferenceResult(
            outputs=dict(zip(self.output_names, outputs)),
            latency_ms=latency,
            memory_used_mb=self.get_memory_usage(),
            backend=self.name
        )
    
    def get_memory_usage(self) -> float:
        try:
            import torch
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            return 0.0
    
    def cleanup(self) -> None:
        self.session = None


# Backend registry
AVAILABLE_BACKENDS = {
    'cpu': ONNXRuntimeCPUBackend,
    'cuda': ONNXRuntimeCUDABackend,
    'rocm': ONNXRuntimeROCmBackend,
    'directml': DirectMLBackend,
    'tensorrt': TensorRTBackend,
}


def get_backend(name: str) -> BaseBackend:
    """Get a backend instance by name"""
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(AVAILABLE_BACKENDS.keys())}")
    return AVAILABLE_BACKENDS[name]()


def detect_available_backends() -> List[str]:
    """Detect which backends are available on this system"""
    available = ['cpu']  # CPU always available
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in providers:
            available.append('cuda')
        if 'ROCMExecutionProvider' in providers:
            available.append('rocm')
        if 'DmlExecutionProvider' in providers:
            available.append('directml')
        if 'TensorrtExecutionProvider' in providers:
            available.append('tensorrt')
    except ImportError:
        pass
    
    return available
