from torch import nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from identity import Identity
from mlp import MLP
from settransformer import SetTransformer
from simpleview.simpleview import MVModel
from dgcnn import DGCNN
from pointnet import PointNet


BACKBONES = ['idt', 'mlp', 'sab', 'isab', 'simpleview','dgcnn', 'pointnet']

class Backbone(nn.Module):
    def __init__(self, backbone_type, d_in, d_out, **kwargs):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        if backbone_type == 'idt':
            self.backbone = Identity()
        elif backbone_type == 'mlp':
            self.backbone = MLP(d_in=self.d_in, d_out=self.d_out, **kwargs)
        elif backbone_type in ['sab', 'isab']:
            # TODO: Override output dim for convenience, could be done better later
            self.d_out = 256
            self.backbone = SetTransformer(type_=backbone_type, d_in=self.d_in, d_out=self.d_out, **kwargs)
        elif backbone_type == 'simpleview':
            self.backbone = MVModel()
            self.d_out = self.backbone.out_dim
        elif backbone_type == 'dgcnn':
            self.backbone = DGCNN()
            self.d_out = 1024
        elif backbone_type == 'pointnet':
            self.backbone = PointNet()
            self.d_out = 512
        else:
            raise ValueError(f'Backbone type {backbone_type} is not implemented!')

    def forward(self, x):
        return self.backbone(x)
