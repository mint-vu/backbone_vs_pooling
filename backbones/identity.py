from torch import nn


class Identity(nn.Module):
    def __init__(self, d_in=None, d_out=None, act_f=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation_function = act_f

    def forward(self, x):
        return x
