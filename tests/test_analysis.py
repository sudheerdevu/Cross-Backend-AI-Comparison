"""
Tests for Cross-Backend AI Comparison

Comprehensive test suite for backend implementations, analysis,
and benchmarking utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os


# ============ Analysis Module Tests ============

class TestBackendResult:
    """Tests for BackendResult dataclass"""
    
    def test_creation(self):
        """Test BackendResult creation"""
        from src.analysis import BackendResult
        
        result = BackendResult(
            backend_name="test-backend",
            model_name="resnet50",
            batch_size=1,
            latency_mean=10.5,
            latency_median=10.0,
            latency_std=1.2,
            latency_min=8.0,
            latency_max=15.0,
            latency_p90=12.0,
            latency_p95=13.0,
            latency_p99=14.5,
            throughput_samples_per_sec=95.0,
            throughput_batches_per_sec=95.0,
        )
        
        assert result.backend_name == "test-backend"
        assert result.model_name == "resnet50"
        assert result.latency_mean == 10.5
    
    def test_to_dict(self):
        """Test BackendResult serialization"""
        from src.analysis import BackendResult
        
        result = BackendResult(
            backend_name="cuda",
            model_name="bert",
            batch_size=8,
            latency_mean=5.0,
            latency_median=4.8,
            latency_std=0.5,
            latency_min=4.0,
            latency_max=6.5,
            latency_p90=5.5,
            latency_p95=6.0,
            latency_p99=6.3,
            throughput_samples_per_sec=1600.0,
            throughput_batches_per_sec=200.0,
            gpu_utilization=85.0,
        )
        
        data = result.to_dict()
        
        assert data['backend_name'] == "cuda"
        assert data['latency']['mean_ms'] == 5.0
        assert data['throughput']['samples_per_sec'] == 1600.0
        assert data['resources']['gpu_utilization'] == 85.0


class TestBenchmarkAnalyzer:
    """Tests for BenchmarkAnalyzer class"""
    
    def test_add_result(self):
        """Test adding results to analyzer"""
        from src.analysis import BenchmarkAnalyzer, BackendResult
        
        analyzer = BenchmarkAnalyzer()
        
        result = BackendResult(
            backend_name="rocm",
            model_name="vgg16",
            batch_size=4,
            latency_mean=20.0,
            latency_median=19.5,
            latency_std=2.0,
            latency_min=16.0,
            latency_max=25.0,
            latency_p90=22.0,
            latency_p95=23.0,
            latency_p99=24.0,
            throughput_samples_per_sec=200.0,
            throughput_batches_per_sec=50.0,
        )
        
        analyzer.add_result(result)
        
        assert "vgg16_4" in analyzer.results
        assert len(analyzer.results["vgg16_4"]) == 1
    
    def test_compare_backends(self):
        """Test backend comparison"""
        from src.analysis import BenchmarkAnalyzer, BackendResult
        
        analyzer = BenchmarkAnalyzer()
        
        # Add results for two backends
        results = [
            BackendResult(
                backend_name="cuda",
                model_name="resnet50",
                batch_size=1,
                latency_mean=8.0,
                latency_median=7.8,
                latency_std=0.8,
                latency_min=6.5,
                latency_max=10.0,
                latency_p90=9.0,
                latency_p95=9.5,
                latency_p99=9.8,
                throughput_samples_per_sec=125.0,
                throughput_batches_per_sec=125.0,
            ),
            BackendResult(
                backend_name="cpu",
                model_name="resnet50",
                batch_size=1,
                latency_mean=45.0,
                latency_median=44.0,
                latency_std=3.0,
                latency_min=40.0,
                latency_max=55.0,
                latency_p90=48.0,
                latency_p95=50.0,
                latency_p99=53.0,
                throughput_samples_per_sec=22.0,
                throughput_batches_per_sec=22.0,
            ),
        ]
        
        analyzer.add_results(results)
        
        comparison = analyzer.compare_backends("resnet50", 1)
        
        assert comparison is not None
        assert comparison.fastest_backend == "cuda"
        assert comparison.slowest_backend == "cpu"
        assert comparison.speedup_vs_slowest > 5.0
        assert comparison.latency_ranking == ["cuda", "cpu"]
    
    def test_generate_markdown_report(self):
        """Test markdown report generation"""
        from src.analysis import BenchmarkAnalyzer, BackendResult
        
        analyzer = BenchmarkAnalyzer()
        
        results = [
            BackendResult(
                backend_name="rocm",
                model_name="mobilenet",
                batch_size=1,
                latency_mean=5.0,
                latency_median=4.8,
                latency_std=0.5,
                latency_min=4.0,
                latency_max=6.5,
                latency_p90=5.5,
                latency_p95=6.0,
                latency_p99=6.2,
                throughput_samples_per_sec=200.0,
                throughput_batches_per_sec=200.0,
            ),
            BackendResult(
                backend_name="tensorrt",
                model_name="mobilenet",
                batch_size=1,
                latency_mean=3.0,
                latency_median=2.9,
                latency_std=0.3,
                latency_min=2.5,
                latency_max=4.0,
                latency_p90=3.3,
                latency_p95=3.5,
                latency_p99=3.8,
                throughput_samples_per_sec=333.0,
                throughput_batches_per_sec=333.0,
            ),
        ]
        
        analyzer.add_results(results)
        
        report = analyzer.generate_report("mobilenet", 1, format="markdown")
        
        assert "# Backend Comparison Report" in report
        assert "mobilenet" in report
        assert "rocm" in report
        assert "tensorrt" in report
    
    def test_export_and_load(self):
        """Test results export and loading"""
        from src.analysis import BenchmarkAnalyzer, BackendResult
        
        analyzer = BenchmarkAnalyzer()
        
        result = BackendResult(
            backend_name="openvino",
            model_name="yolov5",
            batch_size=1,
            latency_mean=12.0,
            latency_median=11.5,
            latency_std=1.5,
            latency_min=9.0,
            latency_max=16.0,
            latency_p90=13.5,
            latency_p95=14.5,
            latency_p99=15.5,
            throughput_samples_per_sec=83.0,
            throughput_batches_per_sec=83.0,
        )
        
        analyzer.add_result(result)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            analyzer.export_results(temp_path)
            
            # Load and verify
            loaded = BenchmarkAnalyzer.load_results(temp_path)
            
            assert "yolov5_1" in loaded.results
            assert loaded.results["yolov5_1"][0].backend_name == "openvino"
        finally:
            os.unlink(temp_path)


# ============ Backend Tests ============

class TestBackends:
    """Tests for backend implementations"""
    
    def test_base_backend_abstract(self):
        """Test that BaseBackend is abstract"""
        from src.backends import BaseBackend
        
        with pytest.raises(TypeError):
            BaseBackend("test")
    
    def test_inference_result(self):
        """Test InferenceResult dataclass"""
        from src.backends import InferenceResult
        
        result = InferenceResult(
            outputs={"output": np.array([1, 2, 3])},
            latency_ms=5.5,
            memory_used_mb=1024.0,
            backend="test"
        )
        
        assert result.latency_ms == 5.5
        assert result.backend == "test"
    
    def test_get_backend(self):
        """Test backend factory"""
        from src.backends import get_backend, AVAILABLE_BACKENDS
        
        # CPU backend should always work
        cpu = get_backend("cpu")
        assert cpu.name == "onnxruntime-cpu"
        
        # Unknown backend should raise
        with pytest.raises(ValueError):
            get_backend("nonexistent")
    
    @patch('src.backends.ort')
    def test_detect_backends(self, mock_ort):
        """Test backend detection"""
        from src.backends import detect_available_backends
        
        mock_ort.get_available_providers.return_value = [
            'CPUExecutionProvider',
            'CUDAExecutionProvider'
        ]
        
        available = detect_available_backends()
        
        assert 'cpu' in available


# ============ Convenience Function Tests ============

class TestConvenienceFunctions:
    """Tests for module-level convenience functions"""
    
    def test_compare_backends_function(self):
        """Test compare_backends convenience function"""
        from src.analysis import compare_backends, BackendResult
        
        results = [
            BackendResult(
                backend_name="a",
                model_name="test",
                batch_size=1,
                latency_mean=10.0,
                latency_median=10.0,
                latency_std=1.0,
                latency_min=8.0,
                latency_max=12.0,
                latency_p90=11.0,
                latency_p95=11.5,
                latency_p99=11.8,
                throughput_samples_per_sec=100.0,
                throughput_batches_per_sec=100.0,
            ),
            BackendResult(
                backend_name="b",
                model_name="test",
                batch_size=1,
                latency_mean=20.0,
                latency_median=20.0,
                latency_std=2.0,
                latency_min=16.0,
                latency_max=24.0,
                latency_p90=22.0,
                latency_p95=23.0,
                latency_p99=23.5,
                throughput_samples_per_sec=50.0,
                throughput_batches_per_sec=50.0,
            ),
        ]
        
        comparison = compare_backends(results, "test", 1)
        
        assert comparison.fastest_backend == "a"
    
    def test_generate_comparison_report_function(self):
        """Test report generation convenience function"""
        from src.analysis import generate_comparison_report, BackendResult
        
        results = [
            BackendResult(
                backend_name="x",
                model_name="model",
                batch_size=1,
                latency_mean=5.0,
                latency_median=5.0,
                latency_std=0.5,
                latency_min=4.0,
                latency_max=6.0,
                latency_p90=5.5,
                latency_p95=5.8,
                latency_p99=5.9,
                throughput_samples_per_sec=200.0,
                throughput_batches_per_sec=200.0,
            ),
            BackendResult(
                backend_name="y",
                model_name="model",
                batch_size=1,
                latency_mean=15.0,
                latency_median=15.0,
                latency_std=1.5,
                latency_min=12.0,
                latency_max=18.0,
                latency_p90=16.5,
                latency_p95=17.0,
                latency_p99=17.5,
                throughput_samples_per_sec=67.0,
                throughput_batches_per_sec=67.0,
            ),
        ]
        
        report = generate_comparison_report(results, "model", 1)
        
        assert "Backend Comparison Report" in report


# ============ Runner Tests ============

class TestRunner:
    """Tests for benchmark runner"""
    
    @patch('src.runner.get_backend')
    def test_benchmark_runner_creation(self, mock_get_backend):
        """Test BenchmarkRunner creation"""
        # This would test runner functionality with mocked backends
        pass


# ============ Integration Tests ============

class TestIntegration:
    """Integration tests requiring actual backends"""
    
    @pytest.mark.skipif(True, reason="Requires actual ONNX model")
    def test_full_benchmark_flow(self):
        """Test complete benchmark workflow"""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
