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
from settransformer import PMA


class Pooling(nn.Module):
    def __init__(self, pooling_type, d_in, **kwargs):
        super().__init__()
        self.d_in = d_in

        if pooling_type == 'COV':
            self.d_out = d_in*(d_in+1)/2
            self.pooling = CovPool(d_in=self.d_in, d_out=self.d_out, **kwargs)
        elif pooling_type == 'GAP':
            self.d_out = d_in
            self.pooling = GAP()
        elif pooling_type == 'GMEAN':
            self.d_out = d_in
            self.pooling = GMean(**kwargs)
        elif pooling_type == 'MAX':
            self.d_out = d_in
            self.pooling = MaxPool()
        elif pooling_type == 'NMax':
            self.d_out = d_in
            self.pooling = NMax(**kwargs)
        # elif pooling_type == 'PMA':
        #     # TODO: set d_out, should be (batch_size, num_seeds, d_in)
        #     self.pooling = PMA(dim=self.d_in, **kwargs)
        # TODO: Add FSPool, FPSWE, LPSWE
        else:
            raise ValueError(f'Pooling type {pooling_type} is not implemented!')

    def forward(self, x):
        return self.pooling(x)
