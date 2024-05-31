from torch import nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from identity import Identity
from mlp import MLP
from settransformer import SetTransformer
from dgcnn import DGCNN
from pointnet.pointnet import PointNet
from pointnet.pointnet2 import PointNet2
from curvenet.curvenet import CurveNet
from pointmlp import pointMLP
from pointnext.pointnext import PointNextEncoder
from pointtransformer import PointTransformerV2

BACKBONES = ['idt', 'mlp', 'sab', 'isab', 'dgcnn', 'pointnet', 'pointnet2', 'curvenet', 'pointmlp', 'pointnext', 'pointtransformer']

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
            self.d_out = 256
            self.backbone = SetTransformer(type_=backbone_type, d_in=self.d_in, d_out=self.d_out, **kwargs)
        elif backbone_type == 'dgcnn':
            self.backbone = DGCNN()
            self.d_out = 1024
        elif backbone_type == 'pointnet':
            self.backbone = PointNet()
            self.d_out = 1024
        elif backbone_type == 'pointnet2':
            self.backbone = PointNet2()
            self.d_out = 128
        elif backbone_type == 'curvenet':
            self.backbone = CurveNet()
            self.d_out = 1024
        elif backbone_type == 'pointmlp':
            self.backbone = pointMLP()
            self.d_out = 64
        elif backbone_type == 'pointnext':
            self.backbone = PointNextEncoder()
            self.d_out = 512
        elif backbone_type == 'pointtransformer':
            self.backbone = PointTransformerV2()
            self.d_out = 3
        else:
            raise ValueError(f'Backbone type {backbone_type} is not implemented!')

    def forward(self, x):
        return self.backbone(x)
