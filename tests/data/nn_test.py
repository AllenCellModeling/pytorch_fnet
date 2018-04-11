import torch
import pdb

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
