import torch
import json

from models.yolo import load_model
from backends.pytorch import PyTorchBackend
from benchmarks.stats import build_result, compare_outputs
from backends.torch_compile import TorchCompileBackend
from backends.onnxruntime import ONNXRuntimeBackend
from benchmarks.runner import BenchmarkRunner

warmup_iterations = 10
benchmark_iterations = 100

batch_size = 1
image_size = 640


def main():
    model = load_model()

    model.model.eval()
    model.model.to("cpu")

    runner = BenchmarkRunner(warmup_iterations=warmup_iterations, benchmark_iterations=benchmark_iterations)

    input_tensor = torch.randn(batch_size, 3, image_size, image_size)

    backend = PyTorchBackend(model.model)

    times = runner.run(backend, input_tensor)
    reference_output = backend.infer(input_tensor)

    result = build_result(backend="PyTorch", device="CPU", input_shape=input_tensor.shape, times=times)

    print("Backend: PyTorch")
    print("Device: CPU")
    print("Input: {batch_size} × 3 × {image_size} × {image_size}")
    print(f"Mean: {result['mean_ms']:.2f} ms")
    print(f"P50:  {result['p50_ms']:.2f} ms")
    print(f"P95:  {result['p95_ms']:.2f} ms")
    print(f"P99:  {result['p99_ms']:.2f} ms")
    print(f"Min:  {result['min_ms']:.2f} ms")
    print(f"Max:  {result['max_ms']:.2f} ms")

    compiled_backend = TorchCompileBackend(model.model)

    compiled_times = runner.run(compiled_backend, input_tensor)
    compiled_output = compiled_backend.infer(input_tensor)

    compiled_accuracy = compare_outputs(reference_output, compiled_output)

    compiled_result = build_result(backend="torch.compile", device="CPU", input_shape=input_tensor.shape, times=compiled_times)

    print("\nBackend: torch.compile")
    print("Device: CPU")
    print("Input: {batch_size} × 3 × {image_size} × {image_size}")
    print(f"Mean: {compiled_result['mean_ms']:.2f} ms")
    print(f"P50:  {compiled_result['p50_ms']:.2f} ms")
    print(f"P95:  {compiled_result['p95_ms']:.2f} ms")
    print(f"P99:  {compiled_result['p99_ms']:.2f} ms")
    print(f"Min:  {compiled_result['min_ms']:.2f} ms")
    print(f"Max:  {compiled_result['max_ms']:.2f} ms")

    onnx_backend = ONNXRuntimeBackend("yolo11n.onnx")

    onnx_times = runner.run(onnx_backend, input_tensor.numpy())
    onnx_output = onnx_backend.infer(input_tensor.numpy())

    onnx_accuracy = compare_outputs(reference_output, onnx_output[0])

    onnx_result = build_result(backend="ONNX Runtime", device="CPU", input_shape=input_tensor.shape, times=onnx_times)

    print("\nBackend: ONNX Runtime")
    print("Device: CPU")
    print("Input: {batch_size} × 3 × {image_size} × {image_size}")
    print(f"Mean: {onnx_result['mean_ms']:.2f} ms")
    print(f"P50:  {onnx_result['p50_ms']:.2f} ms")
    print(f"P95:  {onnx_result['p95_ms']:.2f} ms")
    print(f"P99:  {onnx_result['p99_ms']:.2f} ms")
    print(f"Min:  {onnx_result['min_ms']:.2f} ms")
    print(f"Max:  {onnx_result['max_ms']:.2f} ms")

    compiled_result["accuracy"] = compiled_accuracy
    onnx_result["accuracy"] = onnx_accuracy

    results = [result, compiled_result, onnx_result]

    with open("results/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
