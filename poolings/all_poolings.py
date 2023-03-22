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
from attention import GlobalMultiHeadAttentionPooling as GMHAP
from attention import MultiResolutionMultiHeadAttentionPooling as MMHA
from janossy import Janossy
from bipartite import ApproxRepSet as ARS


POOLINGS = ['cov', 'gap', 'gmean', 'max', 'nmax', 'pma','fpswe','lpswe','lpwe','fspool','janossy','gmha','mmha','bpt']

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
            self.pooling = NMax(**kwargs)
            self.d_out = self.d_in * self.pooling.top_k
        elif pooling_type == 'pma':
            self.d_out = d_in
            self.pooling = PMA(dim=self.d_in, **kwargs)
        elif pooling_type == 'fpswe':
            self.d_out=1024
            self.pooling = FPSWE(d_in=d_in, num_ref_points=1024, **kwargs)
        elif pooling_type == 'lpswe':
            self.d_out=1024
            self.pooling = LPSWE(d_in=d_in, num_ref_points=1024, **kwargs)
            
        elif pooling_type == 'fspool':
            self.d_out=d_in
            self.pooling = FSPool(in_channels=d_in, n_pieces=1024, **kwargs)
            
        elif pooling_type == 'gmha':
            self.pooling = GMHAP(d_in)
            self.d_out=8*d_in

        elif pooling_type == 'mmha':
            self.pooling = MMHA(d_in)
            self.d_out=8*d_in

        elif pooling_type == 'bpt':
            self.pooling = ARS(d_in,d_in,5)
            self.d_out=self.pooling.n_hidden_sets

        elif pooling_type == 'lpwe':
            self.d_out = 9
            self.pooling = OTKernel(in_dim=self.d_in,out_size=3, **kwargs)
        else:
            raise ValueError(f'Pooling type {pooling_type} is not implemented!')

    def forward(self, x):
        return self.pooling(x)
