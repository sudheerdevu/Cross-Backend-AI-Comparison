# Cross-Backend AI Comparison Study

> Systematic comparison of AI inference across multiple hardware backends and execution providers.

## 🎯 Purpose

Compare AI inference performance across:
- **CPU**: OpenMP, Intel MKL, ARM Compute Library
- **NVIDIA GPU**: CUDA, TensorRT
- **AMD GPU**: ROCm, MIGraphX, DirectML
- **NPU/Accelerators**: DirectML (NPU), OpenVINO

## 📊 Study Methodology

```
┌─────────────────────────────────────────────────────────────┐
│                 Study Methodology                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Models  │──▶│  Test       │──▶│  Results    │           │
│  │ Suite   │   │  Harness    │   │  Analysis   │           │
│  └─────────┘   └──────┬──────┘   └─────────────┘           │
│                       │                                     │
│         ┌─────────────┼─────────────┐                       │
│         ▼             ▼             ▼                       │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│   │ Backend  │ │ Backend  │ │ Backend  │  ...              │
│   │    A     │ │    B     │ │    C     │                   │
│   └──────────┘ └──────────┘ └──────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
Cross-Backend-AI-Comparison/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── backends/
│   │   ├── base.py           # Backend interface
│   │   ├── cpu.py            # CPU backend
│   │   ├── cuda.py           # CUDA backend
│   │   ├── rocm.py           # ROCm backend
│   │   └── directml.py       # DirectML backend
│   ├── runner.py             # Test runner
│   ├── analysis.py           # Results analysis
│   └── visualization.py      # Charts and reports
├── models/
│   ├── vision/               # ResNet, EfficientNet, etc.
│   ├── nlp/                  # BERT, GPT-2, etc.
│   └── audio/                # Whisper, etc.
├── configs/
│   ├── quick_test.yaml
│   └── full_comparison.yaml
└── results/
    └── .gitkeep
```

## 🚀 Quick Start

```python
from cross_backend import ComparisonRunner

runner = ComparisonRunner(
    model="models/resnet50.onnx",
    backends=["cpu", "cuda", "rocm", "directml"],
    batch_sizes=[1, 4, 16, 32]
)

results = runner.run()
results.generate_report("comparison_report.html")
```

## 📈 Example Results

| Backend | ResNet50 (B=1) | ResNet50 (B=32) | BERT-base (B=1) |
|---------|----------------|-----------------|-----------------|
| CPU (OpenMP) | 45 ms | 450 ms | 120 ms |
| CUDA (RTX 4090) | 2.1 ms | 8.5 ms | 4.2 ms |
| ROCm (RX 7900) | 2.8 ms | 11.2 ms | 5.8 ms |
| DirectML (NPU) | 8.5 ms | 35 ms | 22 ms |

## 🔧 Metrics Collected

- Latency (P50, P90, P99, P99.9)
- Throughput (inferences/sec)
- Power consumption (where available)
- GPU memory usage
- First inference time (cold start)
- Warm inference time

## 📚 Models Tested

### Vision
- ResNet-50, ResNet-101
- EfficientNet-B0 to B7
- MobileNetV3
- Vision Transformer (ViT)

### NLP
- BERT-base, BERT-large
- GPT-2
- T5 encoder

### Audio
- Whisper tiny/base/small

## License

MIT
