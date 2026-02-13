"""
Analysis Module

Provides statistical analysis and comparison tools for benchmark results.
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BackendResult:
    """Results from a single backend benchmark"""
    backend_name: str
    model_name: str
    batch_size: int
    
    # Latency metrics (ms)
    latency_mean: float
    latency_median: float
    latency_std: float
    latency_min: float
    latency_max: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    
    # Throughput metrics
    throughput_samples_per_sec: float
    throughput_batches_per_sec: float
    
    # Resource utilization
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    cpu_utilization: Optional[float] = None
    
    # Metadata
    iterations: int = 100
    warmup_iterations: int = 10
    device_name: str = ""
    framework_version: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'backend_name': self.backend_name,
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'latency': {
                'mean_ms': self.latency_mean,
                'median_ms': self.latency_median,
                'std_ms': self.latency_std,
                'min_ms': self.latency_min,
                'max_ms': self.latency_max,
                'p90_ms': self.latency_p90,
                'p95_ms': self.latency_p95,
                'p99_ms': self.latency_p99,
            },
            'throughput': {
                'samples_per_sec': self.throughput_samples_per_sec,
                'batches_per_sec': self.throughput_batches_per_sec,
            },
            'resources': {
                'gpu_utilization': self.gpu_utilization,
                'gpu_memory_used_mb': self.gpu_memory_used_mb,
                'gpu_memory_total_mb': self.gpu_memory_total_mb,
                'cpu_utilization': self.cpu_utilization,
            },
            'metadata': {
                'iterations': self.iterations,
                'warmup_iterations': self.warmup_iterations,
                'device_name': self.device_name,
                'framework_version': self.framework_version,
                'timestamp': self.timestamp,
            }
        }


@dataclass
class ComparisonResult:
    """Results from comparing multiple backends"""
    model_name: str
    batch_size: int
    backends: Dict[str, BackendResult]
    
    # Relative performance
    fastest_backend: str
    slowest_backend: str
    speedup_vs_slowest: float  # fastest / slowest
    
    # Rankings
    latency_ranking: List[str]  # Best to worst
    throughput_ranking: List[str]
    
    # Statistical significance
    significant_differences: List[Tuple[str, str, float]]  # (backend1, backend2, p_value)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'backends': {k: v.to_dict() for k, v in self.backends.items()},
            'analysis': {
                'fastest_backend': self.fastest_backend,
                'slowest_backend': self.slowest_backend,
                'speedup': self.speedup_vs_slowest,
                'latency_ranking': self.latency_ranking,
                'throughput_ranking': self.throughput_ranking,
            }
        }


class BenchmarkAnalyzer:
    """
    Analyzes and compares benchmark results across backends.
    """
    
    def __init__(self):
        self.results: Dict[str, List[BackendResult]] = {}  # model -> results
    
    def add_result(self, result: BackendResult) -> None:
        """Add a benchmark result"""
        key = f"{result.model_name}_{result.batch_size}"
        if key not in self.results:
            self.results[key] = []
        self.results[key].append(result)
    
    def add_results(self, results: List[BackendResult]) -> None:
        """Add multiple benchmark results"""
        for result in results:
            self.add_result(result)
    
    def compare_backends(self,
                         model_name: str,
                         batch_size: int) -> Optional[ComparisonResult]:
        """
        Compare all backends for a specific model and batch size.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size used
            
        Returns:
            ComparisonResult or None if insufficient data
        """
        key = f"{model_name}_{batch_size}"
        results = self.results.get(key, [])
        
        if len(results) < 2:
            logger.warning(f"Need at least 2 backends to compare for {key}")
            return None
        
        # Create backend dict
        backends = {r.backend_name: r for r in results}
        
        # Find fastest and slowest
        sorted_by_latency = sorted(results, key=lambda r: r.latency_mean)
        fastest = sorted_by_latency[0]
        slowest = sorted_by_latency[-1]
        
        # Calculate speedup
        speedup = slowest.latency_mean / fastest.latency_mean if fastest.latency_mean > 0 else 1.0
        
        # Create rankings
        latency_ranking = [r.backend_name for r in sorted_by_latency]
        
        sorted_by_throughput = sorted(results, key=lambda r: r.throughput_samples_per_sec, reverse=True)
        throughput_ranking = [r.backend_name for r in sorted_by_throughput]
        
        # Check statistical significance (simplified)
        significant_diffs = self._check_significance(results)
        
        return ComparisonResult(
            model_name=model_name,
            batch_size=batch_size,
            backends=backends,
            fastest_backend=fastest.backend_name,
            slowest_backend=slowest.backend_name,
            speedup_vs_slowest=speedup,
            latency_ranking=latency_ranking,
            throughput_ranking=throughput_ranking,
            significant_differences=significant_diffs,
        )
    
    def _check_significance(self, 
                            results: List[BackendResult],
                            threshold: float = 0.05) -> List[Tuple[str, str, float]]:
        """
        Check for statistically significant differences between backends.
        
        This is a simplified implementation - in practice you'd want
        proper statistical tests with multiple runs.
        """
        significant = []
        
        for i, r1 in enumerate(results):
            for r2 in results[i+1:]:
                # Simple heuristic: significant if means differ by > 2 std devs
                combined_std = (r1.latency_std + r2.latency_std) / 2
                if combined_std > 0:
                    diff = abs(r1.latency_mean - r2.latency_mean)
                    if diff > 2 * combined_std:
                        # Approximate p-value
                        p_value = 0.01 if diff > 3 * combined_std else 0.05
                        significant.append((r1.backend_name, r2.backend_name, p_value))
        
        return significant
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        summary = {
            'total_benchmarks': sum(len(v) for v in self.results.values()),
            'models': list(set(r.model_name for results in self.results.values() for r in results)),
            'backends': list(set(r.backend_name for results in self.results.values() for r in results)),
            'comparisons': {},
        }
        
        # Add comparison for each model/batch combo
        for key in self.results:
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                model_name, batch_size = parts[0], int(parts[1])
                comparison = self.compare_backends(model_name, batch_size)
                if comparison:
                    summary['comparisons'][key] = {
                        'fastest': comparison.fastest_backend,
                        'speedup': comparison.speedup_vs_slowest,
                    }
        
        return summary
    
    def generate_report(self,
                        model_name: str,
                        batch_size: int,
                        format: str = 'markdown') -> str:
        """
        Generate a comparison report.
        
        Args:
            model_name: Model to report on
            batch_size: Batch size
            format: Report format ('markdown', 'html')
            
        Returns:
            Report string
        """
        comparison = self.compare_backends(model_name, batch_size)
        if not comparison:
            return "Insufficient data for comparison"
        
        if format == 'html':
            return self._generate_html_report(comparison)
        else:
            return self._generate_markdown_report(comparison)
    
    def _generate_markdown_report(self, comparison: ComparisonResult) -> str:
        """Generate markdown report"""
        lines = [
            f"# Backend Comparison Report",
            f"",
            f"**Model:** {comparison.model_name}",
            f"**Batch Size:** {comparison.batch_size}",
            f"**Backends Compared:** {len(comparison.backends)}",
            f"",
            f"## Summary",
            f"",
            f"- **Fastest Backend:** {comparison.fastest_backend}",
            f"- **Speedup vs Slowest:** {comparison.speedup_vs_slowest:.2f}x",
            f"",
            f"## Latency Comparison",
            f"",
            f"| Backend | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |",
            f"|---------|-----------|----------|----------|----------|",
        ]
        
        for name in comparison.latency_ranking:
            r = comparison.backends[name]
            lines.append(
                f"| {name} | {r.latency_mean:.2f} | {r.latency_median:.2f} | "
                f"{r.latency_p95:.2f} | {r.latency_p99:.2f} |"
            )
        
        lines.extend([
            f"",
            f"## Throughput Comparison",
            f"",
            f"| Backend | Samples/sec | Batches/sec |",
            f"|---------|-------------|-------------|",
        ])
        
        for name in comparison.throughput_ranking:
            r = comparison.backends[name]
            lines.append(
                f"| {name} | {r.throughput_samples_per_sec:.1f} | "
                f"{r.throughput_batches_per_sec:.1f} |"
            )
        
        lines.extend([
            f"",
            f"## Rankings",
            f"",
            f"**Latency (best to worst):** {' > '.join(comparison.latency_ranking)}",
            f"",
            f"**Throughput (best to worst):** {' > '.join(comparison.throughput_ranking)}",
        ])
        
        return "\n".join(lines)
    
    def _generate_html_report(self, comparison: ComparisonResult) -> str:
        """Generate HTML report"""
        # Create table rows for latency
        latency_rows = ""
        for i, name in enumerate(comparison.latency_ranking):
            r = comparison.backends[name]
            rank_class = 'best' if i == 0 else ('worst' if i == len(comparison.latency_ranking) - 1 else '')
            latency_rows += f"""
            <tr class="{rank_class}">
                <td>{i+1}</td>
                <td>{name}</td>
                <td>{r.latency_mean:.2f}</td>
                <td>{r.latency_median:.2f}</td>
                <td>{r.latency_p95:.2f}</td>
                <td>{r.latency_p99:.2f}</td>
            </tr>
            """
        
        # Create table rows for throughput
        throughput_rows = ""
        for i, name in enumerate(comparison.throughput_ranking):
            r = comparison.backends[name]
            throughput_rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{name}</td>
                <td>{r.throughput_samples_per_sec:.1f}</td>
                <td>{r.throughput_batches_per_sec:.1f}</td>
            </tr>
            """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backend Comparison: {comparison.model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr.best {{ background: #c8e6c9; }}
        tr.worst {{ background: #ffcdd2; }}
        tr:hover {{ background: #f5f5f5; }}
        .speedup {{ font-size: 1.5em; color: #4CAF50; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backend Comparison Report</h1>
        
        <div class="summary">
            <p><strong>Model:</strong> {comparison.model_name}</p>
            <p><strong>Batch Size:</strong> {comparison.batch_size}</p>
            <p><strong>Fastest Backend:</strong> {comparison.fastest_backend}</p>
            <p><strong>Speedup:</strong> <span class="speedup">{comparison.speedup_vs_slowest:.2f}x</span></p>
        </div>
        
        <h2>Latency Comparison</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Backend</th>
                <th>Mean (ms)</th>
                <th>P50 (ms)</th>
                <th>P95 (ms)</th>
                <th>P99 (ms)</th>
            </tr>
            {latency_rows}
        </table>
        
        <h2>Throughput Comparison</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Backend</th>
                <th>Samples/sec</th>
                <th>Batches/sec</th>
            </tr>
            {throughput_rows}
        </table>
    </div>
</body>
</html>"""
        
        return html
    
    def export_results(self, filepath: str) -> None:
        """Export all results to JSON file"""
        data = {
            'results': {k: [r.to_dict() for r in v] for k, v in self.results.items()},
            'summary': self.get_summary(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_results(cls, filepath: str) -> 'BenchmarkAnalyzer':
        """Load results from JSON file"""
        analyzer = cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for key, results in data.get('results', {}).items():
            for r in results:
                lat = r.get('latency', {})
                tp = r.get('throughput', {})
                res = r.get('resources', {})
                meta = r.get('metadata', {})
                
                result = BackendResult(
                    backend_name=r['backend_name'],
                    model_name=r['model_name'],
                    batch_size=r['batch_size'],
                    latency_mean=lat.get('mean_ms', 0),
                    latency_median=lat.get('median_ms', 0),
                    latency_std=lat.get('std_ms', 0),
                    latency_min=lat.get('min_ms', 0),
                    latency_max=lat.get('max_ms', 0),
                    latency_p90=lat.get('p90_ms', 0),
                    latency_p95=lat.get('p95_ms', 0),
                    latency_p99=lat.get('p99_ms', 0),
                    throughput_samples_per_sec=tp.get('samples_per_sec', 0),
                    throughput_batches_per_sec=tp.get('batches_per_sec', 0),
                    gpu_utilization=res.get('gpu_utilization'),
                    gpu_memory_used_mb=res.get('gpu_memory_used_mb'),
                    iterations=meta.get('iterations', 100),
                    device_name=meta.get('device_name', ''),
                )
                analyzer.add_result(result)
        
        return analyzer


# ============ Convenience Functions ============

def compare_backends(results: List[BackendResult],
                     model_name: str,
                     batch_size: int) -> Optional[ComparisonResult]:
    """
    Compare backends from a list of results.
    
    Args:
        results: List of benchmark results
        model_name: Model name to compare
        batch_size: Batch size to compare
        
    Returns:
        ComparisonResult or None
    """
    analyzer = BenchmarkAnalyzer()
    analyzer.add_results(results)
    return analyzer.compare_backends(model_name, batch_size)


def generate_comparison_report(results: List[BackendResult],
                                model_name: str,
                                batch_size: int,
                                format: str = 'markdown') -> str:
    """
    Generate a comparison report from results.
    
    Args:
        results: List of benchmark results
        model_name: Model name
        batch_size: Batch size
        format: Report format
        
    Returns:
        Report string
    """
    analyzer = BenchmarkAnalyzer()
    analyzer.add_results(results)
    return analyzer.generate_report(model_name, batch_size, format)
