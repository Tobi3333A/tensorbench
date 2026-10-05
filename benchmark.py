import torch

from models.yolo import load_model
from backends.pytorch import PyTorchBackend
from benchmarks.stats import summarize_latencies


def main():
    model = load_model()

    model.model.eval()
    model.model.to("cpu")

    input_tensor = torch.randn(1, 3, 640, 640)

    backend = PyTorchBackend(model.model)

    times = backend.benchmark(input_tensor, iterations=100)
    stats = summarize_latencies(times)

    print("Backend: PyTorch")
    print("Device: CPU")
    print("Input: 1 × 3 × 640 × 640")
    print(f"Mean: {stats['mean_ms']:.2f} ms")
    print(f"P50:  {stats['p50_ms']:.2f} ms")
    print(f"P95:  {stats['p95_ms']:.2f} ms")
    print(f"P99:  {stats['p99_ms']:.2f} ms")
    print(f"Min:  {stats['min_ms']:.2f} ms")
    print(f"Max:  {stats['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()

