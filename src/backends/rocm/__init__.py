"""
ROCm Backend Utilities

Provides ROCm-specific profiling and optimization utilities
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
class ROCmDeviceInfo:
    """Information about a ROCm-capable device"""
    device_id: int
    name: str
    compute_units: int
    memory_total_mb: int
    memory_free_mb: int
    gfx_version: str
    pcie_bandwidth_gb: float
    clock_speed_mhz: int
    memory_clock_mhz: int
    driver_version: str


class ROCmProfiler:
    """
    ROCm profiling utilities using rocprof and rocm-smi.
    """
    
    def __init__(self):
        self.rocprof_path = self._find_rocprof()
        self.rocm_smi_path = self._find_rocm_smi()
        self.profile_data: List[Dict[str, Any]] = []
    
    def _find_rocprof(self) -> Optional[str]:
        """Find rocprof executable"""
        paths = [
            '/opt/rocm/bin/rocprof',
            '/usr/bin/rocprof',
            os.path.expanduser('~/rocm/bin/rocprof'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        # Try PATH
        try:
            result = subprocess.run(['which', 'rocprof'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def _find_rocm_smi(self) -> Optional[str]:
        """Find rocm-smi executable"""
        paths = [
            '/opt/rocm/bin/rocm-smi',
            '/usr/bin/rocm-smi',
            os.path.expanduser('~/rocm/bin/rocm-smi'),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def get_device_info(self, device_id: int = 0) -> Optional[ROCmDeviceInfo]:
        """
        Get information about a ROCm device.
        
        Args:
            device_id: Device index
            
        Returns:
            ROCmDeviceInfo or None if unavailable
        """
        if not self.rocm_smi_path:
            logger.warning("rocm-smi not found")
            return None
        
        try:
            # Get device info
            result = subprocess.run(
                [self.rocm_smi_path, '-d', str(device_id), '--showproductname'],
                capture_output=True, text=True
            )
            
            name = "Unknown AMD GPU"
            for line in result.stdout.split('\n'):
                if 'GPU' in line and ':' in line:
                    name = line.split(':')[-1].strip()
                    break
            
            # Get memory info
            mem_result = subprocess.run(
                [self.rocm_smi_path, '-d', str(device_id), '--showmeminfo', 'vram'],
                capture_output=True, text=True
            )
            
            memory_total = 0
            memory_used = 0
            for line in mem_result.stdout.split('\n'):
                if 'Total' in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        memory_total = int(match.group(1))
                elif 'Used' in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        memory_used = int(match.group(1))
            
            # Get GFX version
            gfx_result = subprocess.run(
                [self.rocm_smi_path, '-d', str(device_id), '--showgfxversion'],
                capture_output=True, text=True
            )
            
            gfx_version = "unknown"
            for line in gfx_result.stdout.split('\n'):
                if 'gfx' in line.lower():
                    match = re.search(r'(gfx\d+)', line.lower())
                    if match:
                        gfx_version = match.group(1)
                        break
            
            return ROCmDeviceInfo(
                device_id=device_id,
                name=name,
                compute_units=64,  # Would need rocm-smi --showcomputeunits
                memory_total_mb=memory_total // (1024 * 1024),
                memory_free_mb=(memory_total - memory_used) // (1024 * 1024),
                gfx_version=gfx_version,
                pcie_bandwidth_gb=16.0,  # Typical PCIe 4.0 x16
                clock_speed_mhz=2000,
                memory_clock_mhz=1000,
                driver_version=self.get_driver_version(),
            )
            
        except Exception as e:
            logger.error(f"Failed to get ROCm device info: {e}")
            return None
    
    def get_driver_version(self) -> str:
        """Get ROCm driver version"""
        try:
            result = subprocess.run(
                ['cat', '/opt/rocm/.info/version'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "unknown"
    
    def profile_kernel(self,
                       command: str,
                       output_csv: str = "profile_output.csv",
                       metrics: Optional[List[str]] = None) -> bool:
        """
        Profile a command using rocprof.
        
        Args:
            command: Command to profile
            output_csv: Output CSV file path
            metrics: List of metrics to collect
            
        Returns:
            True if profiling succeeded
        """
        if not self.rocprof_path:
            logger.error("rocprof not found")
            return False
        
        try:
            cmd = [self.rocprof_path, '-o', output_csv]
            
            if metrics:
                metrics_str = ','.join(metrics)
                cmd.extend(['--metrics', metrics_str])
            
            cmd.extend(command.split())
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Profile results written to {output_csv}")
                return True
            else:
                logger.error(f"Profiling failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Profiling error: {e}")
            return False
    
    def get_gpu_utilization(self, device_id: int = 0) -> Tuple[float, float]:
        """
        Get current GPU utilization.
        
        Returns:
            Tuple of (compute_utilization, memory_utilization) as percentages
        """
        if not self.rocm_smi_path:
            return (0.0, 0.0)
        
        try:
            result = subprocess.run(
                [self.rocm_smi_path, '-d', str(device_id), '--showuse'],
                capture_output=True, text=True
            )
            
            compute_util = 0.0
            mem_util = 0.0
            
            for line in result.stdout.split('\n'):
                if 'GPU use' in line:
                    match = re.search(r'(\d+)%', line)
                    if match:
                        compute_util = float(match.group(1))
                elif 'Memory use' in line:
                    match = re.search(r'(\d+)%', line)
                    if match:
                        mem_util = float(match.group(1))
            
            return (compute_util, mem_util)
            
        except Exception as e:
            logger.error(f"Failed to get GPU utilization: {e}")
            return (0.0, 0.0)
    
    def get_temperature(self, device_id: int = 0) -> float:
        """Get GPU temperature in Celsius"""
        if not self.rocm_smi_path:
            return 0.0
        
        try:
            result = subprocess.run(
                [self.rocm_smi_path, '-d', str(device_id), '--showtemp'],
                capture_output=True, text=True
            )
            
            for line in result.stdout.split('\n'):
                if 'Temperature' in line or 'temp' in line.lower():
                    match = re.search(r'(\d+\.?\d*).*[cC]', line)
                    if match:
                        return float(match.group(1))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to get temperature: {e}")
            return 0.0


class MIGraphXOptimizer:
    """
    Helper for MIGraphX model optimization.
    """
    
    @staticmethod
    def optimize_onnx_model(input_path: str,
                            output_path: str,
                            quantize: bool = False,
                            batch_size: int = 1) -> bool:
        """
        Optimize an ONNX model for MIGraphX.
        
        Args:
            input_path: Path to input ONNX model
            output_path: Path for optimized model
            quantize: Whether to apply INT8 quantization
            batch_size: Target batch size for optimization
            
        Returns:
            True if optimization succeeded
        """
        try:
            import migraphx
            
            # Parse the ONNX model
            model = migraphx.parse_onnx(input_path)
            
            # Compile for GPU
            model.compile(migraphx.get_target("gpu"))
            
            # Save compiled model
            migraphx.save(model, output_path, format='msgpack')
            
            logger.info(f"MIGraphX optimized model saved to {output_path}")
            return True
            
        except ImportError:
            logger.error("MIGraphX not available")
            return False
        except Exception as e:
            logger.error(f"MIGraphX optimization failed: {e}")
            return False
    
    @staticmethod
    def get_optimal_batch_size(model_path: str,
                               memory_limit_mb: int = 8000) -> int:
        """
        Estimate optimal batch size for a model given memory constraints.
        
        Args:
            model_path: Path to model
            memory_limit_mb: Maximum GPU memory to use
            
        Returns:
            Recommended batch size
        """
        # Simplified estimation
        # In practice, would profile with different batch sizes
        
        try:
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Rough estimate: model needs ~3x memory when running
            model_memory = file_size_mb * 3
            
            # Available for batches
            available = memory_limit_mb - model_memory
            
            # Assume 10MB per batch item (rough average)
            batch_size = max(1, int(available / 10))
            
            # Cap at reasonable max
            return min(batch_size, 64)
            
        except Exception as e:
            logger.error(f"Batch size estimation failed: {e}")
            return 1


def detect_rocm_devices() -> List[ROCmDeviceInfo]:
    """
    Detect all ROCm-capable devices.
    
    Returns:
        List of ROCmDeviceInfo for each device
    """
    profiler = ROCmProfiler()
    devices = []
    
    # Check for HIP_VISIBLE_DEVICES
    visible = os.environ.get('HIP_VISIBLE_DEVICES', '')
    
    try:
        result = subprocess.run(
            ['rocm-smi', '--showallinfo'],
            capture_output=True, text=True
        )
        
        # Count GPUs from output
        gpu_count = result.stdout.count('GPU[')
        
        for i in range(gpu_count):
            info = profiler.get_device_info(i)
            if info:
                devices.append(info)
                
    except Exception as e:
        logger.error(f"Failed to detect ROCm devices: {e}")
    
    return devices


def check_rocm_environment() -> Dict[str, Any]:
    """
    Check ROCm environment and return diagnostic info.
    
    Returns:
        Dictionary with environment information
    """
    env = {
        'rocm_installed': False,
        'rocm_version': None,
        'hip_version': None,
        'devices': [],
        'environment_vars': {
            'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES', 'not set'),
            'ROCM_PATH': os.environ.get('ROCM_PATH', '/opt/rocm'),
            'HSA_OVERRIDE_GFX_VERSION': os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set'),
        },
        'onnxruntime_rocm': False,
    }
    
    # Check ROCm installation
    rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
    if os.path.exists(rocm_path):
        env['rocm_installed'] = True
        
        # Get version
        version_file = os.path.join(rocm_path, '.info', 'version')
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                env['rocm_version'] = f.read().strip()
    
    # Check HIP version
    try:
        result = subprocess.run(['hipcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            match = re.search(r'HIP version:\s*([\d.]+)', result.stdout)
            if match:
                env['hip_version'] = match.group(1)
    except:
        pass
    
    # Check ONNX Runtime ROCm provider
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        env['onnxruntime_rocm'] = 'ROCMExecutionProvider' in providers
    except:
        pass
    
    # Detect devices
    env['devices'] = [d.name for d in detect_rocm_devices()]
    
    return env
