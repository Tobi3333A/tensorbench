import torch


class PyTorchBackend:
    def __init__(self, model):
        self.model = model

    def warmup(self, input_tensor, iterations=10):
        with torch.inference_mode():
            for _ in range(iterations):
                self.model(input_tensor)

    def infer(self, input_tensor):
        with torch.inference_mode():
            return self.model(input_tensor)
        