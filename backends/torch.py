import time

import torch

class PyTorchBackend:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def warmup(self, input_tensor, iterations=10):
        with torch.inference_mode():
            for _ in range(iterations):
                self.model(input_tensor)

    def infer(self, input_tensor):
        with torch.inference_mode():
            return self.model(input_tensor)

    def benchmark(self, input_tensor, iterations=100):
        self.warmup(input_tensor)

        times = []

        with torch.inference_mode():
            for _ in range(iterations):
                start = time.perf_counter()

                self.model(input_tensor)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)

        return times
