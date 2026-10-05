import time

import onnxruntime as ort


class ONNXRuntimeBackend:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

    def warmup(self, input_tensor, iterations=10):
        for _ in range(iterations):
            self.session.run(
                None,
                {self.input_name: input_tensor},
            )

    def infer(self, input_tensor):
        return self.session.run(
            None,
            {self.input_name: input_tensor},
        )

    def benchmark(self, input_tensor, iterations=100):
        self.warmup(input_tensor)

        times = []

        for _ in range(iterations):
            start = time.perf_counter()

            self.session.run(
                None,
                {self.input_name: input_tensor},
            )

            end = time.perf_counter()

            times.append((end - start) * 1000)

        return times
