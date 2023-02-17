from torch import nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..'))

from cov import CovPool
from gap import GAP
from gmean import GMean
from max import MaxPool
from nmax import NMax
from fspool import FSPool
from fpswe import FPSWE
from lpswe import LPSWE
from lpwe import OTKernel
from attention_layers import PMA


POOLINGS = ['gap', 'gmean', 'max', 'nmax', 'pma','fpswe','lpswe','fspool']

class Pooling(nn.Module):
    def __init__(self, pooling_type, d_in, **kwargs):
        super().__init__()
        self.d_in = d_in

        if pooling_type == 'cov':
            self.d_out = d_in*(d_in+1)//2
            self.pooling = CovPool(d_in=self.d_in, d_out=self.d_out, **kwargs)
        elif pooling_type == 'gap':
            self.d_out = d_in
            self.pooling = GAP()
        elif pooling_type == 'gmean':
            self.d_out = d_in
            self.pooling = GMean(**kwargs)
        elif pooling_type == 'max':
            self.d_out = d_in
            self.pooling = MaxPool()
        elif pooling_type == 'nmax':
            self.d_out = d_in
            self.pooling = NMax(**kwargs)
        elif pooling_type == 'pma':
            self.d_out = d_in
            self.pooling = PMA(dim=self.d_in, **kwargs)
        elif pooling_type == 'fpswe':
            self.d_out=d_in
            self.pooling = FPSWE(d_in=d_in, num_ref_points=1024, **kwargs)
        elif pooling_type == 'lpswe':
            self.d_out=d_in
            self.pooling = LPSWE(d_in=d_in, num_ref_points=1024, **kwargs)
            
        elif pooling_type == 'fspool':
            self.d_out=d_in
            self.pooling = FSPool(in_channels=d_in, **kwargs)

        elif pooling_type == 'lpwe':
            self.pooling = OTKernel(in_dim=self.d_in, **kwargs)
        else:
            raise ValueError(f'Pooling type {pooling_type} is not implemented!')

    def forward(self, x):
        return self.pooling(x)
