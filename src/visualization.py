#!/usr/bin/env python3
"""
Cross-Backend AI Comparison - Visualization

Generate charts and visual comparisons of backend performance.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


def load_results(path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(path) as f:
        return json.load(f)


def plot_latency_comparison(results: List[Dict], 
                           output_path: str = "latency_comparison.png") -> None:
    """Create bar chart comparing latencies across backends."""
    # Group by batch size
    by_batch: Dict[int, Dict[str, float]] = {}
    
    for r in results:
        bs = r['batch_size']
        if bs not in by_batch:
            by_batch[bs] = {}
        by_batch[bs][r['backend']] = r['latency_ms']
    
    batch_sizes = sorted(by_batch.keys())
    backends = list(set(b for bs in by_batch.values() for b in bs.keys()))
    
    x = np.arange(len(batch_sizes))
    width = 0.8 / len(backends)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(backends)))
    
    for i, backend in enumerate(backends):
        values = [by_batch[bs].get(backend, 0) for bs in batch_sizes]
        offset = (i - len(backends)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=backend.upper(), color=colors[i])
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Inference Latency by Backend and Batch Size')
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_throughput_comparison(results: List[Dict],
                               output_path: str = "throughput_comparison.png") -> None:
    """Create throughput comparison chart."""
    by_batch: Dict[int, Dict[str, float]] = {}
    
    for r in results:
        bs = r['batch_size']
        if bs not in by_batch:
            by_batch[bs] = {}
        by_batch[bs][r['backend']] = r['throughput']
    
    batch_sizes = sorted(by_batch.keys())
    backends = list(set(b for bs in by_batch.values() for b in bs.keys()))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(backends)))
    
    for i, backend in enumerate(backends):
        values = [by_batch[bs].get(backend, 0) for bs in batch_sizes]
        ax.plot(batch_sizes, values, 'o-', label=backend.upper(), 
                color=colors[i], linewidth=2, markersize=8)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (inferences/sec)')
    ax.set_title('Inference Throughput by Backend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_percentiles(results: List[Dict],
                             batch_size: int = 1,
                             output_path: str = "latency_percentiles.png") -> None:
    """Create percentile comparison for a specific batch size."""
    filtered = [r for r in results if r['batch_size'] == batch_size]
    
    if not filtered:
        print(f"No results for batch_size={batch_size}")
        return
    
    backends = [r['backend'] for r in filtered]
    p50 = [r['latency_p50'] for r in filtered]
    p90 = [r['latency_p90'] for r in filtered]
    p99 = [r['latency_p99'] for r in filtered]
    
    x = np.arange(len(backends))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, p50, width, label='P50', color='#2ecc71')
    ax.bar(x, p90, width, label='P90', color='#f39c12')
    ax.bar(x + width, p99, width, label='P99', color='#e74c3c')
    
    ax.set_xlabel('Backend')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Latency Percentiles (Batch Size = {batch_size})')
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backends])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_first_inference_overhead(results: List[Dict],
                                  output_path: str = "first_inference.png") -> None:
    """Compare first inference (cold start) vs steady state."""
    # Use batch_size=1 for comparison
    filtered = [r for r in results if r['batch_size'] == 1]
    
    if not filtered:
        return
    
    backends = [r['backend'] for r in filtered]
    first = [r['first_inference_ms'] for r in filtered]
    steady = [r['latency_ms'] for r in filtered]
    
    x = np.arange(len(backends))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, first, width, label='First Inference', color='#e74c3c')
    ax.bar(x + width/2, steady, width, label='Steady State', color='#2ecc71')
    
    ax.set_xlabel('Backend')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Cold Start vs Steady State Latency')
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in backends])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage overhead labels
    for i, (f, s) in enumerate(zip(first, steady)):
        if s > 0:
            overhead = ((f - s) / s) * 100
            ax.annotate(f'+{overhead:.0f}%', 
                       (i - width/2, f), 
                       textcoords="offset points",
                       xytext=(0, 5), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_speedup_heatmap(results: List[Dict],
                         baseline: str = 'cpu',
                         output_path: str = "speedup_heatmap.png") -> None:
    """Create heatmap showing speedup relative to baseline."""
    # Organize data
    by_batch: Dict[int, Dict[str, float]] = {}
    
    for r in results:
        bs = r['batch_size']
        if bs not in by_batch:
            by_batch[bs] = {}
        by_batch[bs][r['backend']] = r['latency_ms']
    
    batch_sizes = sorted(by_batch.keys())
    backends = [b for b in set(r['backend'] for r in results) if b != baseline]
    
    # Calculate speedups
    speedups = np.zeros((len(backends), len(batch_sizes)))
    
    for i, backend in enumerate(backends):
        for j, bs in enumerate(batch_sizes):
            baseline_lat = by_batch.get(bs, {}).get(baseline, 1)
            backend_lat = by_batch.get(bs, {}).get(backend, baseline_lat)
            speedups[i, j] = baseline_lat / backend_lat if backend_lat > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(speedups, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
    
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(backends)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels([b.upper() for b in backends])
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Backend')
    ax.set_title(f'Speedup vs {baseline.upper()} (>1 = faster)')
    
    # Add text annotations
    for i in range(len(backends)):
        for j in range(len(batch_sizes)):
            ax.text(j, i, f'{speedups[i, j]:.2f}x',
                   ha='center', va='center', color='black', fontsize=10)
    
    plt.colorbar(im, label='Speedup')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_html_report(results: List[Dict],
                         output_path: str = "report.html") -> None:
    """Generate interactive HTML report."""
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Cross-Backend AI Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .chart { width: 100%; height: 400px; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .fastest { background-color: #d4edda; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Cross-Backend AI Comparison Report</h1>
    
    <h2>Results Table</h2>
    <table>
        <tr>
            <th>Backend</th>
            <th>Batch Size</th>
            <th>Latency (ms)</th>
            <th>P99 (ms)</th>
            <th>Throughput (inf/s)</th>
        </tr>
'''
    
    # Find fastest for each batch size
    by_batch = {}
    for r in results:
        bs = r['batch_size']
        if bs not in by_batch or r['latency_ms'] < by_batch[bs]['latency_ms']:
            by_batch[bs] = r
    
    for r in results:
        is_fastest = by_batch.get(r['batch_size'], {}).get('backend') == r['backend']
        row_class = 'fastest' if is_fastest else ''
        html += f'''        <tr class="{row_class}">
            <td>{r['backend'].upper()}</td>
            <td>{r['batch_size']}</td>
            <td>{r['latency_ms']:.2f}</td>
            <td>{r['latency_p99']:.2f}</td>
            <td>{r['throughput']:.1f}</td>
        </tr>
'''
    
    html += '''    </table>
    
    <h2>Latency Comparison</h2>
    <div id="latencyChart" class="chart"></div>
    
    <h2>Throughput Scaling</h2>
    <div id="throughputChart" class="chart"></div>
    
    <script>
'''
    
    # Build Plotly data
    by_backend = {}
    for r in results:
        b = r['backend']
        if b not in by_backend:
            by_backend[b] = {'batch_sizes': [], 'latencies': [], 'throughputs': []}
        by_backend[b]['batch_sizes'].append(r['batch_size'])
        by_backend[b]['latencies'].append(r['latency_ms'])
        by_backend[b]['throughputs'].append(r['throughput'])
    
    # Latency chart
    html += "        var latencyData = [\n"
    for backend, data in by_backend.items():
        html += f'''            {{
                x: {data['batch_sizes']},
                y: {data['latencies']},
                type: 'bar',
                name: '{backend.upper()}'
            }},
'''
    html += '''        ];
        Plotly.newPlot('latencyChart', latencyData, {
            barmode: 'group',
            xaxis: {title: 'Batch Size'},
            yaxis: {title: 'Latency (ms)'}
        });
'''
    
    # Throughput chart
    html += "\n        var throughputData = [\n"
    for backend, data in by_backend.items():
        html += f'''            {{
                x: {data['batch_sizes']},
                y: {data['throughputs']},
                type: 'scatter',
                mode: 'lines+markers',
                name: '{backend.upper()}'
            }},
'''
    html += '''        ];
        Plotly.newPlot('throughputChart', throughputData, {
            xaxis: {title: 'Batch Size'},
            yaxis: {title: 'Throughput (inferences/sec)'}
        });
    </script>
</body>
</html>
'''
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Saved: {output_path}")


def generate_all_visualizations(results_path: str,
                                output_dir: str = "visualizations") -> None:
    """Generate all visualizations from results file."""
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    
    results = load_results(results_path)
    
    plot_latency_comparison(results, str(output / "latency_comparison.png"))
    plot_throughput_comparison(results, str(output / "throughput_comparison.png"))
    plot_latency_percentiles(results, 1, str(output / "latency_percentiles_bs1.png"))
    plot_first_inference_overhead(results, str(output / "first_inference.png"))
    plot_speedup_heatmap(results, 'cpu', str(output / "speedup_heatmap.png"))
    generate_html_report(results, str(output / "report.html"))


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <results.json> [output_dir]")
        sys.exit(1)
    
    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
    
    generate_all_visualizations(results_path, output_dir)
