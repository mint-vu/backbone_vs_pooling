import torch
from torch import nn


class GMean(nn.Module):
    def __init__(self, moment=2):
        super().__init__()
        self.moment = moment
    
    def forward(self, x):
        return torch.mean(x**self.moment, dim=1)**(1/self.moment)