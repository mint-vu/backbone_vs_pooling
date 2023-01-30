import torch
from torch import nn


class NMax(nn.Module):
    def __init__(self, top_k=2):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        return torch.topk(x, self.top_k, dim=1)[0]