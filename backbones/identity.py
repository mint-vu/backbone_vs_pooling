from torch import nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Directly return x
        return x
