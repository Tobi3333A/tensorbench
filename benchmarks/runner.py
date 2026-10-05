import time


class BenchmarkRunner:
    def __init__(self, warmup_iterations=10, benchmark_iterations=100):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def run(self, backend, input_tensor):
        backend.warmup(
            input_tensor,
            iterations=self.warmup_iterations,
        )

        times = []

        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()

            backend.infer(input_tensor)

            end = time.perf_counter()

            times.append((end - start) * 1000)

        return times
