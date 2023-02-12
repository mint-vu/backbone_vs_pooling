from torch import nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from identity import Identity
from mlp import MLP
from settransformer import SAB, ISAB


BACKBONES = ['idt', 'mlp', 'sab', 'isab']

class Backbone(nn.Module):
    def __init__(self, backbone_type, d_in, d_out, **kwargs):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        if backbone_type == 'idt':
            self.backbone = Identity()
        elif backbone_type == 'mlp':
            self.backbone = MLP(d_in=self.d_in, d_out=self.d_out, **kwargs)
        elif backbone_type == 'sab':
            self.backbone == SAB(dim_in=self.din, dim_out=self.d_out, **kwargs)
        elif backbone_type == 'isab':
            self.backbone == ISAB(dim_in=self.din, dim_out=self.d_out, **kwargs)
        else:
            raise ValueError(f'Backbone type {backbone_type} is not implemented!')

    def forward(self, x):
        return self.backbone(x)
