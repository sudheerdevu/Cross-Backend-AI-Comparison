"""
OpenVINO Backend Utilities

Provides OpenVINO-specific profiling and optimization utilities
for AI inference benchmarking.
"""

import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpenVINODeviceInfo:
    """Information about an OpenVINO device"""
    device_name: str
    full_device_name: str
    device_type: str  # CPU, GPU, NPU, etc.
    supported_properties: List[str]
    optimization_capabilities: List[str]


class OpenVINOBackend:
    """
    OpenVINO inference backend with profiling support.
    """
    
    def __init__(self):
        self._core = None
        self._model = None
        self._compiled_model = None
        self._infer_request = None
        self._init_core()
    
    def _init_core(self) -> None:
        """Initialize OpenVINO runtime core"""
        try:
            from openvino.runtime import Core
            self._core = Core()
        except ImportError:
            logger.error("OpenVINO not available - install with 'pip install openvino'")
    
    @property
    def core(self):
        return self._core
    
    def get_available_devices(self) -> List[str]:
        """Get list of available OpenVINO devices"""
        if self._core is None:
            return []
        return self._core.available_devices
    
    def get_device_info(self, device: str = "CPU") -> Optional[OpenVINODeviceInfo]:
        """
        Get information about an OpenVINO device.
        
        Args:
            device: Device name (CPU, GPU, NPU, etc.)
            
        Returns:
            OpenVINODeviceInfo or None if unavailable
        """
        if self._core is None:
            return None
        
        try:
            full_name = self._core.get_property(device, "FULL_DEVICE_NAME")
            
            # Get supported properties
            supported_props = []
            try:
                for prop in ["OPTIMIZATION_CAPABILITIES", "NUM_STREAMS", 
                            "INFERENCE_NUM_THREADS", "PERFORMANCE_HINT"]:
                    try:
                        self._core.get_property(device, prop)
                        supported_props.append(prop)
                    except:
                        pass
            except:
                pass
            
            # Get optimization capabilities
            opt_caps = []
            try:
                caps = self._core.get_property(device, "OPTIMIZATION_CAPABILITIES")
                opt_caps = list(caps) if caps else []
            except:
                pass
            
            return OpenVINODeviceInfo(
                device_name=device,
                full_device_name=full_name,
                device_type=device.split('.')[0],
                supported_properties=supported_props,
                optimization_capabilities=opt_caps,
            )
            
        except Exception as e:
            logger.error(f"Failed to get OpenVINO device info: {e}")
            return None
    
    def load_model(self,
                   model_path: str,
                   device: str = "CPU",
                   config: Optional[Dict[str, str]] = None) -> bool:
        """
        Load and compile a model.
        
        Args:
            model_path: Path to ONNX or OpenVINO IR model
            device: Target device
            config: Device configuration options
            
        Returns:
            True if loading succeeded
        """
        if self._core is None:
            return False
        
        try:
            # Read model
            self._model = self._core.read_model(model_path)
            
            # Compile for device
            compile_config = config or {}
            self._compiled_model = self._core.compile_model(
                self._model,
                device,
                compile_config
            )
            
            # Create infer request
            self._infer_request = self._compiled_model.create_infer_request()
            
            logger.info(f"Model loaded successfully on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        if self._infer_request is None:
            raise RuntimeError("Model not loaded. Call load_model first.")
        
        # Set inputs
        for name, data in inputs.items():
            self._infer_request.set_tensor(name, data)
        
        # Run inference
        self._infer_request.infer()
        
        # Get outputs
        outputs = {}
        for output in self._compiled_model.outputs:
            outputs[output.any_name] = self._infer_request.get_tensor(output).data.copy()
        
        return outputs
    
    def get_latency(self) -> float:
        """Get inference latency in milliseconds"""
        if self._infer_request is None:
            return 0.0
        
        try:
            return self._infer_request.latency / 1000.0  # microseconds to ms
        except:
            return 0.0


class OpenVINOOptimizer:
    """
    Helper for OpenVINO model optimization.
    """
    
    @staticmethod
    def convert_onnx_to_ir(input_path: str,
                           output_path: str,
                           input_shapes: Optional[Dict[str, List[int]]] = None,
                           compress_to_fp16: bool = True) -> bool:
        """
        Convert ONNX model to OpenVINO IR format.
        
        Args:
            input_path: Path to ONNX model
            output_path: Path for output IR model (without extension)
            input_shapes: Static input shapes (optional)
            compress_to_fp16: Compress weights to FP16
            
        Returns:
            True if conversion succeeded
        """
        try:
            from openvino.tools import mo
            
            args = [input_path, '-o', output_path]
            
            if compress_to_fp16:
                args.append('--compress_to_fp16')
            
            if input_shapes:
                shape_str = ','.join(
                    f"{name}:{list(shape)}" 
                    for name, shape in input_shapes.items()
                )
                args.extend(['--input_shape', shape_str])
            
            mo.convert_model(*args)
            
            logger.info(f"Converted to OpenVINO IR: {output_path}")
            return True
            
        except ImportError:
            logger.error("OpenVINO Model Optimizer not available")
            return False
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    
    @staticmethod
    def apply_nncf_quantization(model_path: str,
                                output_path: str,
                                calibration_data: Optional[List] = None,
                                preset: str = "performance") -> bool:
        """
        Apply Neural Network Compression Framework (NNCF) quantization.
        
        Args:
            model_path: Path to model (ONNX or IR)
            output_path: Path for quantized model
            calibration_data: Calibration dataset
            preset: Quantization preset ('performance' or 'accuracy')
            
        Returns:
            True if quantization succeeded
        """
        try:
            import nncf
            from openvino.runtime import Core
            
            core = Core()
            model = core.read_model(model_path)
            
            # Create calibration dataset if not provided
            if calibration_data is None:
                logger.warning("No calibration data - using random data")
                calibration_data = []
            
            # Apply quantization
            quantized_model = nncf.quantize(
                model,
                calibration_dataset=nncf.Dataset(calibration_data),
                preset=nncf.QuantizationPreset.PERFORMANCE if preset == "performance" 
                       else nncf.QuantizationPreset.MIXED,
            )
            
            # Save quantized model
            from openvino.runtime import serialize
            serialize(quantized_model, output_path)
            
            logger.info(f"Quantized model saved to {output_path}")
            return True
            
        except ImportError:
            logger.error("NNCF not available - install with 'pip install nncf'")
            return False
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False
    
    @staticmethod
    def get_optimal_config(device: str,
                           hint: str = "latency") -> Dict[str, str]:
        """
        Get optimal OpenVINO configuration for a device.
        
        Args:
            device: Target device (CPU, GPU, etc.)
            hint: Performance hint ('latency' or 'throughput')
            
        Returns:
            Configuration dictionary
        """
        config = {
            "PERFORMANCE_HINT": hint.upper(),
        }
        
        if device.upper() == "CPU":
            config.update({
                "INFERENCE_NUM_THREADS": str(os.cpu_count() or 4),
                "CPU_BIND_THREAD": "YES",
            })
        elif device.upper() == "GPU":
            config.update({
                "GPU_HINT_1": "PERFORMANCE",
            })
        
        return config


def detect_openvino_devices() -> List[OpenVINODeviceInfo]:
    """
    Detect all OpenVINO-capable devices.
    
    Returns:
        List of OpenVINODeviceInfo for each device
    """
    backend = OpenVINOBackend()
    devices = []
    
    for device in backend.get_available_devices():
        info = backend.get_device_info(device)
        if info:
            devices.append(info)
    
    return devices


def check_openvino_environment() -> Dict[str, Any]:
    """
    Check OpenVINO environment and return diagnostic info.
    
    Returns:
        Dictionary with environment information
    """
    env = {
        'openvino_available': False,
        'openvino_version': None,
        'devices': [],
        'nncf_available': False,
        'mo_available': False,
        'onnxruntime_openvino': False,
    }
    
    # Check OpenVINO runtime
    try:
        from openvino.runtime import get_version
        env['openvino_available'] = True
        env['openvino_version'] = get_version()
    except ImportError:
        pass
    
    # Check NNCF
    try:
        import nncf
        env['nncf_available'] = True
    except ImportError:
        pass
    
    # Check Model Optimizer
    try:
        from openvino.tools import mo
        env['mo_available'] = True
    except ImportError:
        pass
    
    # Check ONNX Runtime OpenVINO EP
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        env['onnxruntime_openvino'] = 'OpenVINOExecutionProvider' in providers
    except:
        pass
    
    # Get devices
    for info in detect_openvino_devices():
        env['devices'].append({
            'name': info.device_name,
            'full_name': info.full_device_name,
            'type': info.device_type,
        })
    
    return env


# Convenience function for benchmarking
def run_openvino_benchmark(model_path: str,
                           device: str = "CPU",
                           iterations: int = 100,
                           warmup: int = 10) -> Dict[str, float]:
    """
    Run a simple benchmark using OpenVINO.
    
    Args:
        model_path: Path to model
        device: Target device
        iterations: Number of iterations
        warmup: Warmup iterations
        
    Returns:
        Dictionary with latency statistics
    """
    import time
    import numpy as np
    
    backend = OpenVINOBackend()
    
    if not backend.load_model(model_path, device):
        return {}
    
    # Get input info
    from openvino.runtime import Core
    core = Core()
    model = core.read_model(model_path)
    
    # Create random inputs
    inputs = {}
    for inp in model.inputs:
        shape = list(inp.shape)
        # Replace dynamic dims
        shape = [s if s > 0 else 1 for s in shape]
        inputs[inp.any_name] = np.random.rand(*shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        backend.infer(inputs)
    
    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        backend.infer(inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    
    return {
        'mean_ms': np.mean(latencies),
        'median_ms': np.median(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p90_ms': np.percentile(latencies, 90),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
    }
