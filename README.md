# Tensorbench

Benchmarking toolkit for evaluating ML model performance across inference backends.

## Motivation

Different inference backends can run the same model with different performance. I built tensorbench to compare these differences using the same model, input, and timing setup.

The project currently benchmarks YOLO11n with:

- PyTorch
- torch.compile
- ONNX Runtime

For each backend, tensorbench records several latency statistics and compares the outputs against the PyTorch implementation.

## Benchmark Setup

The benchmark uses:

- Model: YOLO11n
- Device: CPU
- Batch size: 1
- Input size: 640 × 640
- Warmup iterations: 10
- Benchmark iterations: 100

The following latency measurements are recorded:

- Mean
- P50
- P95
- P99
- Minimum
- Maximum

## Results

| Backend | Mean | P50 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 86.73 ms | 82.93 ms | 129.83 ms | 148.48 ms |
| torch.compile | 58.23 ms | 58.24 ms | 65.22 ms | 70.32 ms |
| ONNX Runtime | **41.90 ms** | **40.53 ms** | **46.13 ms** | **73.80 ms** |

ONNX Runtime had the lowest latency across the benchmark, with a P50 of 40.53 ms.

The optimized backends were also compared against the PyTorch output:

| Backend | Mean Absolute Error | Max Absolute Error |
| --- | ---: | ---: |
| torch.compile | 2.61 × 10⁻⁶ | 0.00204 |
| ONNX Runtime | 2.49 × 10⁻⁶ | 0.00134 |

## Project Structure

```text
tensorbench/
├── backends/
│   ├── onnxruntime.py
│   ├── pytorch.py
│   └── torch_compile.py
├── benchmarks/
│   ├── runner.py
│   └── stats.py
├── models/
│   └── yolo.py
├── results/
│   └── benchmark.json
├── benchmark.py
└── README.md
```

## Running

Install the dependencies and run:

```bash
python benchmark.py
```

The benchmark runs each backend and saves the results to:

```text
results/benchmark.json
```

## Why I Built It

I wanted to understand how the same ML model performs when run through different inference backends. Instead of only looking at whether a model works, this project focuses on measuring the latency differences between them and checking that optimized versions still produce similar outputs.
