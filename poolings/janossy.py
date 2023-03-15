import torch
from torch import nn


class Janossy(nn.Module):
    def __init__(self, first_k):
        super().__init__()
        self.k = first_k

    def forward(self, X):
        sorted_indices = X[:, 0].sort()[1]
        X = X[sorted_indices]
        total_num = X.shape[0]
        all_k_ary = torch.combinations(torch.arange(total_num), r=self.k)
        X_jp = []
        for perm in all_k_ary:
            X_jp.append(X[perm])
        X_jp = torch.stack(X_jp, dim=0)

        return X_jp.mean(dim=0)