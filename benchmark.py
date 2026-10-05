import torch

from models.yolo import load_model
from backends.pytorch import PyTorchBackend
from benchmarks.stats import build_result
from backends.torch_compile import TorchCompileBackend
from backends.onnxruntime import ONNXRuntimeBackend
from benchmarks.runner import BenchmarkRunner


def main():
    model = load_model()

    model.model.eval()
    model.model.to("cpu")

    runner = BenchmarkRunner(warmup_iterations=10, benchmark_iterations=100)

    input_tensor = torch.randn(1, 3, 640, 640)

    backend = PyTorchBackend(model.model)

    times = runner.run(backend, input_tensor)

    result = build_result(backend="PyTorch", device="CPU", input_shape=input_tensor.shape, times=times)

    print("Backend: PyTorch")
    print("Device: CPU")
    print("Input: 1 × 3 × 640 × 640")
    print(f"Mean: {result['mean_ms']:.2f} ms")
    print(f"P50:  {result['p50_ms']:.2f} ms")
    print(f"P95:  {result['p95_ms']:.2f} ms")
    print(f"P99:  {result['p99_ms']:.2f} ms")
    print(f"Min:  {result['min_ms']:.2f} ms")
    print(f"Max:  {result['max_ms']:.2f} ms")

    compiled_backend = TorchCompileBackend(model.model)

    compiled_times = runner.run(compiled_backend, input_tensor)

    compiled_result = build_result(backend="torch.compile", device="CPU", input_shape=input_tensor.shape, times=compiled_times)

    print("\nBackend: torch.compile")
    print("Device: CPU")
    print("Input: 1 × 3 × 640 × 640")
    print(f"Mean: {compiled_result['mean_ms']:.2f} ms")
    print(f"P50:  {compiled_result['p50_ms']:.2f} ms")
    print(f"P95:  {compiled_result['p95_ms']:.2f} ms")
    print(f"P99:  {compiled_result['p99_ms']:.2f} ms")
    print(f"Min:  {compiled_result['min_ms']:.2f} ms")
    print(f"Max:  {compiled_result['max_ms']:.2f} ms")

    onnx_backend = ONNXRuntimeBackend("yolo11n.onnx")

    onnx_times = runner.run(onnx_backend, input_tensor.numpy())

    onnx_result = build_result(backend="ONNX Runtime", device="CPU", input_shape=input_tensor.shape, times=onnx_times)

    print("\nBackend: ONNX Runtime")
    print("Device: CPU")
    print("Input: 1 × 3 × 640 × 640")
    print(f"Mean: {onnx_result['mean_ms']:.2f} ms")
    print(f"P50:  {onnx_result['p50_ms']:.2f} ms")
    print(f"P95:  {onnx_result['p95_ms']:.2f} ms")
    print(f"P99:  {onnx_result['p99_ms']:.2f} ms")
    print(f"Min:  {onnx_result['min_ms']:.2f} ms")
    print(f"Max:  {onnx_result['max_ms']:.2f} ms")


if __name__ == "__main__":
    main()
