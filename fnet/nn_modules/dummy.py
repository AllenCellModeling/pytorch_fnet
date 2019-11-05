import torch


class DummyModel(torch.nn.Module):
    def __init__(self, some_param=42):
        super().__init__()
        self.some_param = some_param
        self.network = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1)

    def __call__(self, x):
        return self.network(x)
