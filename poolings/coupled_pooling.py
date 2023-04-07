import torch
from torch import nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from all_poolings import Pooling

class CoupledPooling(nn.Module):
    def __init__(self, pooling_types, d_in, pooling_args=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = 0

        self.poolings = nn.ModuleList()
        for pooling_type in pooling_types:
            if pooling_args is not None and pooling_type in pooling_args:
                kwargs = pooling_args[pooling_type]
            else:
                kwargs = {}
            self.poolings.append(Pooling(pooling_type, d_in, **kwargs))
            self.d_out += self.poolings[-1].d_out
        
    def forward(self, x):
        out = []
        for pooling in self.poolings:
            out.append(pooling(x))
        return torch.cat(out, dim=1)

    