from torch import nn
from identity import Identity
from mlp import MLP


class Backbone(nn.Module):
    def __init__(self, backbone_type='IDT', d_in=None, d_out=None, num_layers=1):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        if backbone_type == 'IDT':
            self.backbone = Identity()
            self.d_out = d_in
        if backbone_type == 'MLP':
            self.backbone = MLP(d_in=self.d_in, d_out=d_out,
                                num_layers=num_layers)
            self.d_out = d_out
        else:
            raise ValueError(f'Backbone type {backbone_type} is not implemented!')

    def forward(self, x):
        return self.backbone(x)
