
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class ApproxRepSet(nn.Module):

    def __init__(self, d, n_hidden_sets, n_elements):
        super(ApproxRepSet, self).__init__()
        self.n_hidden_sets = n_hidden_sets
        self.n_elements = n_elements
        
        self.Wc = Parameter(torch.FloatTensor(d, n_hidden_sets*n_elements))
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-1, 1)

    def forward(self, X):
        t = self.relu(torch.matmul(X, self.Wc))
        t = t.view(t.size()[0], t.size()[1], self.n_elements, self.n_hidden_sets)
        t,_ = torch.max(t, dim=2)
        out = torch.sum(t, dim=1)

        return out