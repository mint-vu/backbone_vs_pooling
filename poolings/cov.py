import torch
from torch import nn


class CovPool(nn.Module):
    def __init__(self, d_in, batch_size=32, lambda_=1e-2):
        super(CovPool,self).__init__()
        self.d_in = d_in
        self.d_out = int(d_in*(d_in+1)/2)
        self.batch_size = batch_size
        self.l = lambda_

    def batch_cov(self, points):
        B, N, D = points.size()
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return bcov  # (B, D, D)

    def forward(self, x):
        c=torch.unique(torch.flatten(self.batch_cov(x) + self.l*torch.eye(self.d_in, device=x.device)))
        
        return c.reshape(self.batch_size,self.d_out)
