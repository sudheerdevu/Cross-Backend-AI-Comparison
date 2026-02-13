# Cross-Backend Benchmarking Guide

## Quick Start

### Installation

```bash
# Base installation
pip install -e .

# With ROCm support
pip install -e ".[rocm]"

# With CUDA support
pip install -e ".[cuda]"
```

### Running Your First Benchmark

```bash
# Run comparison on ResNet50
python src/benchmark_runner.py \
    --model models/resnet50.onnx \
    --backends rocm cuda cpu \
    --iterations 100
```

## Usage Examples

### Compare Latency

```python
from src.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(
    model_path="models/bert-base.onnx",
    backends=["rocm", "cpu"]
)

results = runner.run(iterations=100, warmup=10)

for backend, metrics in results.items():
    print(f"{backend}: {metrics['latency_p50']:.2f}ms")
```

### Batch Size Analysis

```python
for batch_size in [1, 4, 8, 16, 32]:
    results = runner.run_with_batch_size(batch_size)
    print(f"Batch {batch_size}: {results['throughput']:.1f} samples/sec")
```

## Interpreting Results

### Latency Metrics

- **P50**: Median latency (typical case)
- **P99**: Tail latency (worst case)
- **Mean**: Average latency

### Throughput

```
Throughput = Batch Size / Latency
```

## Troubleshooting

### Backend Not Found

```bash
# Check available backends
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### Memory Issues

Use smaller batch sizes or enable memory pool:

```python
session_options.enable_mem_pattern = True
session_options.enable_mem_reuse = True
```
