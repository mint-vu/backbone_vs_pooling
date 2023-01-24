from torch import nn
from cov import CovPool
from gap import GAP
from gmean import GMean
from max import MaxPool
from nmax import NMax

class Pooling(nn.Module):
    def __init__(self, pooling_type='GAP', d_in=1, batch_size=32, moment=2, top_k=2):
        super().__init__()

        if pooling_type == 'COV':
            self.num_outputs = d_in*(d_in+1)/2
            self.pooling = CovPool(d_in=d_in, num_outputs=self.num_outputs, batch_size=batch_size)
        elif pooling_type == 'GAP':
            self.num_outputs = d_in
            self.pooling = GAP()
        elif pooling_type == 'GMEAN':
            self.num_outputs = d_in
            self.pooling = GMean(moment=moment)
        elif pooling_type == 'MAX':
            self.num_outputs = d_in
            self.pooling = MaxPool()
        elif pooling_type == 'NMax':
            self.num_outputs = d_in
            self.pooling = NMax(top_k=top_k)
        else:
            raise ValueError(f'Pooling type {pooling_type} is not implemented!')

    def forward(self, x):
        return self.pooling(x)
