"""
CUDA Backend Utilities

Provides CUDA and TensorRT-specific profiling and optimization utilities
for AI inference benchmarking.
"""

import os
import subprocess
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CUDADeviceInfo:
    """Information about a CUDA-capable device"""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    memory_total_mb: int
    memory_free_mb: int
    multiprocessors: int
    cuda_cores: int
    clock_speed_mhz: int
    memory_clock_mhz: int
    memory_bus_width: int
    driver_version: str
    cuda_version: str


class CUDAProfiler:
    """
    CUDA profiling utilities using NVML and profiling tools.
    """
    
    def __init__(self):
        self._nvml_initialized = False
        self._init_nvml()
    
    def _init_nvml(self) -> None:
        """Initialize NVML for GPU monitoring"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except ImportError:
            logger.warning("pynvml not available - install with 'pip install nvidia-ml-py3'")
        except Exception as e:
            logger.warning(f"NVML initialization failed: {e}")
    
    def get_device_count(self) -> int:
        """Get number of CUDA devices"""
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            pass
        
        try:
            import pynvml
            if self._nvml_initialized:
                return pynvml.nvmlDeviceGetCount()
        except:
            pass
        
        return 0
    
    def get_device_info(self, device_id: int = 0) -> Optional[CUDADeviceInfo]:
        """
        Get information about a CUDA device.
        
        Args:
            device_id: Device index
            
        Returns:
            CUDADeviceInfo or None if unavailable
        """
        try:
            import torch
            
            if device_id >= torch.cuda.device_count():
                return None
            
            props = torch.cuda.get_device_properties(device_id)
            
            # Get memory info
            total_mem = props.total_memory
            free_mem = total_mem - torch.cuda.memory_allocated(device_id)
            
            # Get driver/CUDA version
            driver_version = ""
            cuda_version = ""
            
            try:
                import pynvml
                if self._nvml_initialized:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    cuda_version = str(pynvml.nvmlSystemGetCudaDriverVersion())
            except:
                pass
            
            # Estimate CUDA cores based on architecture
            cuda_cores = self._estimate_cuda_cores(
                props.multi_processor_count,
                props.major,
                props.minor
            )
            
            return CUDADeviceInfo(
                device_id=device_id,
                name=props.name,
                compute_capability=(props.major, props.minor),
                memory_total_mb=total_mem // (1024 * 1024),
                memory_free_mb=free_mem // (1024 * 1024),
                multiprocessors=props.multi_processor_count,
                cuda_cores=cuda_cores,
                clock_speed_mhz=props.processor_count,  # Not accurate, placeholder
                memory_clock_mhz=0,
                memory_bus_width=props.memory_clock_rate,  # Not accurate
                driver_version=driver_version,
                cuda_version=cuda_version,
            )
            
        except ImportError:
            logger.error("PyTorch not available for CUDA device info")
        except Exception as e:
            logger.error(f"Failed to get CUDA device info: {e}")
        
        return None
    
    def _estimate_cuda_cores(self, sm_count: int, major: int, minor: int) -> int:
        """Estimate CUDA cores based on SM count and compute capability"""
        # Cores per SM by architecture
        cores_per_sm = {
            (3, 0): 192,  # Kepler
            (3, 5): 192,
            (5, 0): 128,  # Maxwell
            (5, 2): 128,
            (6, 0): 64,   # Pascal
            (6, 1): 128,
            (7, 0): 64,   # Volta
            (7, 5): 64,   # Turing
            (8, 0): 64,   # Ampere
            (8, 6): 128,
            (8, 9): 128,
            (9, 0): 128,  # Hopper
        }
        
        key = (major, minor)
        if key in cores_per_sm:
            return sm_count * cores_per_sm[key]
        
        # Default estimate for unknown architectures
        return sm_count * 64
    
    def get_gpu_utilization(self, device_id: int = 0) -> Tuple[float, float]:
        """
        Get current GPU utilization.
        
        Returns:
            Tuple of (compute_utilization, memory_utilization) as percentages
        """
        try:
            import pynvml
            
            if not self._nvml_initialized:
                return (0.0, 0.0)
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            return (float(utilization.gpu), float(utilization.memory))
            
        except Exception as e:
            logger.error(f"Failed to get GPU utilization: {e}")
            return (0.0, 0.0)
    
    def get_temperature(self, device_id: int = 0) -> float:
        """Get GPU temperature in Celsius"""
        try:
            import pynvml
            
            if not self._nvml_initialized:
                return 0.0
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return float(temp)
            
        except Exception as e:
            logger.error(f"Failed to get temperature: {e}")
            return 0.0
    
    def get_power_usage(self, device_id: int = 0) -> float:
        """Get GPU power usage in Watts"""
        try:
            import pynvml
            
            if not self._nvml_initialized:
                return 0.0
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
            
            return power / 1000.0
            
        except Exception as e:
            logger.error(f"Failed to get power usage: {e}")
            return 0.0
    
    def synchronize(self, device_id: Optional[int] = None) -> None:
        """Synchronize CUDA device"""
        try:
            import torch
            
            if device_id is not None:
                torch.cuda.set_device(device_id)
            torch.cuda.synchronize()
            
        except ImportError:
            pass
    
    def cleanup(self) -> None:
        """Cleanup NVML resources"""
        try:
            import pynvml
            if self._nvml_initialized:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
        except:
            pass


class TensorRTOptimizer:
    """
    Helper for TensorRT model optimization.
    """
    
    @staticmethod
    def optimize_onnx_model(input_path: str,
                            output_path: str,
                            fp16: bool = True,
                            int8: bool = False,
                            max_batch_size: int = 16,
                            workspace_gb: float = 2.0) -> bool:
        """
        Optimize an ONNX model with TensorRT.
        
        Args:
            input_path: Path to input ONNX model
            output_path: Path for optimized TensorRT engine
            fp16: Enable FP16 precision
            int8: Enable INT8 quantization (requires calibration data)
            max_batch_size: Maximum supported batch size
            workspace_gb: GPU workspace size in GB
            
        Returns:
            True if optimization succeeded
        """
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(TRT_LOGGER)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            
            # Parse ONNX
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            with open(input_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(i)}")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = int(workspace_gb * 1024 * 1024 * 1024)
            
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            
            if int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Set optimization profile for dynamic batch
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                input_tensor = network.get_input(i)
                shape = list(input_tensor.shape)
                
                # Set min/opt/max shapes
                min_shape = [1] + shape[1:]
                opt_shape = [max_batch_size // 2] + shape[1:]
                max_shape = [max_batch_size] + shape[1:]
                
                profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            
            config.add_optimization_profile(profile)
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved to {output_path}")
            return True
            
        except ImportError:
            logger.error("TensorRT not available")
            return False
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            return False
    
    @staticmethod
    def get_trt_version() -> Optional[str]:
        """Get TensorRT version"""
        try:
            import tensorrt as trt
            return trt.__version__
        except:
            return None


def detect_cuda_devices() -> List[CUDADeviceInfo]:
    """
    Detect all CUDA-capable devices.
    
    Returns:
        List of CUDADeviceInfo for each device
    """
    profiler = CUDAProfiler()
    devices = []
    
    device_count = profiler.get_device_count()
    
    for i in range(device_count):
        info = profiler.get_device_info(i)
        if info:
            devices.append(info)
    
    return devices


def check_cuda_environment() -> Dict[str, Any]:
    """
    Check CUDA environment and return diagnostic info.
    
    Returns:
        Dictionary with environment information
    """
    env = {
        'cuda_available': False,
        'cuda_version': None,
        'cudnn_version': None,
        'driver_version': None,
        'devices': [],
        'tensorrt_available': False,
        'tensorrt_version': None,
        'environment_vars': {
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
            'CUDA_HOME': os.environ.get('CUDA_HOME', 'not set'),
        },
        'onnxruntime_cuda': False,
        'onnxruntime_tensorrt': False,
    }
    
    # Check PyTorch CUDA
    try:
        import torch
        env['cuda_available'] = torch.cuda.is_available()
        if env['cuda_available']:
            env['cuda_version'] = torch.version.cuda
            env['cudnn_version'] = str(torch.backends.cudnn.version())
    except ImportError:
        pass
    
    # Check TensorRT
    try:
        import tensorrt as trt
        env['tensorrt_available'] = True
        env['tensorrt_version'] = trt.__version__
    except ImportError:
        pass
    
    # Check ONNX Runtime providers
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        env['onnxruntime_cuda'] = 'CUDAExecutionProvider' in providers
        env['onnxruntime_tensorrt'] = 'TensorrtExecutionProvider' in providers
    except:
        pass
    
    # Get device info
    for info in detect_cuda_devices():
        env['devices'].append({
            'name': info.name,
            'memory_mb': info.memory_total_mb,
            'compute_capability': f"{info.compute_capability[0]}.{info.compute_capability[1]}",
        })
        if not env['driver_version']:
            env['driver_version'] = info.driver_version
    
    return env
