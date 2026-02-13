# Cross-Backend Comparison Architecture

## Overview

This framework provides unified benchmarking across multiple AI inference backends.

## Supported Backends

| Backend | Provider | GPU Support |
|---------|----------|-------------|
| ROCm | AMD | Yes (RDNA/CDNA) |
| CUDA | NVIDIA | Yes |
| OpenVINO | Intel | CPU/iGPU |
| ONNX Runtime | Microsoft | CPU/GPU |

## Architecture

```
┌─────────────────────────────────────────────┐
│              Benchmark Runner               │
├──────────────┬──────────────┬───────────────┤
│   ROCm EP    │   CUDA EP    │  OpenVINO EP  │
├──────────────┴──────────────┴───────────────┤
│           ONNX Runtime Core                 │
└─────────────────────────────────────────────┘
```

## Components

### 1. Benchmark Runner (`src/benchmark_runner.py`)

- Orchestrates test execution
- Manages warmup and measurement iterations
- Collects timing statistics

### 2. Backend Adapters (`src/backends/`)

Each backend has an adapter implementing:
- `initialize()`: Load model and create session
- `run_inference()`: Execute single inference
- `get_metrics()`: Return backend-specific metrics

### 3. Result Analyzer (`src/analyzer.py`)

- Aggregates results across backends
- Generates comparison reports
- Statistical significance testing

## Configuration

```yaml
# config/benchmark.yaml
backends:
  - name: rocm
    device_id: 0
  - name: cpu
    threads: 4

metrics:
  - latency_p50
  - latency_p99
  - throughput
```

## Data Flow

1. Load model (ONNX format)
2. Initialize all configured backends
3. Run warmup iterations
4. Execute timed iterations
5. Collect and aggregate metrics
6. Generate comparison report
