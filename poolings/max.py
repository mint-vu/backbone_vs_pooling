import torch
from torch import nn


class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.max(x, dim=1)[0]