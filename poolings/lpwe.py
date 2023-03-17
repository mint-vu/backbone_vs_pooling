# Source: https://github.com/claying/OTK

import math
import random
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim


class OTKernel(nn.Module):
    def __init__(self, in_dim, out_size, heads=1, eps=0.1, max_iter=100,
                 log_domain=False, position_encoding=None, position_sigma=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_size = out_size
        self.heads = heads
        self.eps = eps
        self.max_iter = max_iter

        self.weight = nn.Parameter(
            torch.Tensor(heads, out_size, in_dim))

        self.log_domain = log_domain
        self.position_encoding = position_encoding
        self.position_sigma = position_sigma

        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.out_size)
        for w in self.parameters():
            w.data.uniform_(-stdv, stdv)

    def get_position_filter(self, input, out_size):
        if input.ndim == 4:
            in_size1 = input.shape[1]
            in_size2 = input.shape[2]
            out_size = int(math.sqrt(out_size))
            if self.position_encoding is None:
                return self.position_encoding
            elif self.position_encoding == "gaussian":
                sigma = self.position_sigma
                a1 = torch.arange(1., in_size1 + 1.).view(-1, 1) / in_size1
                a2 = torch.arange(1., in_size2 + 1.).view(-1, 1) / in_size2
                b = torch.arange(1., out_size + 1.).view(1, -1) / out_size
                position_filter1 = torch.exp(-((a1 - b) / sigma) ** 2)
                position_filter2 = torch.exp(-((a2 - b) / sigma) ** 2)
                position_filter = position_filter1.view(
                    in_size1, 1, out_size, 1) * position_filter2.view(
                    1, in_size2, 1, out_size)
            if self.weight.is_cuda:
                position_filter = position_filter.cuda()
            return position_filter.reshape(1, 1, in_size1 * in_size2, out_size * out_size)
        in_size = input.shape[1]
        if self.position_encoding is None:
            return self.position_encoding
        elif self.position_encoding == "gaussian":
            # sigma = 1. / out_size
            sigma = self.position_sigma
            a = torch.arange(0., in_size).view(-1, 1) / in_size
            b = torch.arange(0., out_size).view(1, -1) / out_size
            position_filter = torch.exp(-((a - b) / sigma) ** 2)
        elif self.position_encoding == "hard":
            # sigma = 1. / out_size
            sigma = self.position_sigma
            a = torch.arange(0., in_size).view(-1, 1) / in_size
            b = torch.arange(0., out_size).view(1, -1) / out_size
            position_filter = torch.abs(a - b) < sigma
            position_filter = position_filter.float()
        else:
            raise ValueError("Unrecognizied position encoding")
        if self.weight.is_cuda:
            position_filter = position_filter.cuda()
        position_filter = position_filter.view(1, 1, in_size, out_size)
        return position_filter

    def get_attn(self, input, mask=None, position_filter=None):
        """Compute the attention weight using Sinkhorn OT
        input: batch_size x in_size x in_dim
        mask: batch_size x in_size
        self.weight: heads x out_size x in_dim
        output: batch_size x (out_size x heads) x in_size
        """
        return multihead_attn(
            input, self.weight, mask=mask, eps=self.eps,
            max_iter=self.max_iter, log_domain=self.log_domain,
            position_filter=position_filter)

    def forward(self, input, mask=None):
        """
        input: batch_size x in_size x in_dim
        output: batch_size x out_size x (heads x in_dim)
        """
        batch_size = input.shape[0]
        position_filter = self.get_position_filter(input, self.out_size)
        in_ndim = input.ndim
        if in_ndim == 4:
            input = input.view(batch_size, -1, self.in_dim)
        attn_weight = self.get_attn(input, mask, position_filter)
        # attn_weight: batch_size x out_size x heads x in_size

        output = torch.bmm(
            attn_weight.view(batch_size, self.out_size * self.heads, -1), input)
        if in_ndim == 4:
            out_size = int(math.sqrt(self.out_size))
            output = output.reshape(batch_size, out_size, out_size, -1)
        else:
            output = output.reshape(batch_size, self.out_size, -1)
        B,N,D = output.shape
        return output.view(B,N*D)

    def unsup_train(self, input, wb=False, inplace=True, use_cuda=False):
        """K-meeans for learning parameters
        input: n_samples x in_size x in_dim
        weight: heads x out_size x in_dim
        """
        input_normalized = normalize(input, inplace=inplace)
        block_size = int(1e9) // (input.shape[1] * input.shape[2] * 4)
        print("Starting Wasserstein K-means")
        weight = wasserstein_kmeans(
            input_normalized, self.heads, self.out_size, eps=self.eps,
            block_size=block_size, wb=wb, log_domain=self.log_domain, use_cuda=use_cuda)
        self.weight.data.copy_(weight)

    def random_sample(self, input):
        idx = torch.randint(0, input.shape[0], (1,))
        self.weight.data.copy_(input[idx].view_as(self.weight))

class Linear(nn.Linear):
    def forward(self, input):
        bias = self.bias
        if bias is not None and hasattr(self, 'scale_bias') and self.scale_bias is not None:
            bias = self.scale_bias * bias
        out = torch.nn.functional.linear(input, self.weight, bias)
        return out

    def fit(self, Xtr, ytr, criterion, reg=0.0, epochs=100, optimizer=None, use_cuda=False):
        if optimizer is None:
            optimizer = optim.LBFGS(self.parameters(), lr=1.0, history_size=10)
        if self.bias is not None:
            scale_bias = (Xtr ** 2).mean(-1).sqrt().mean().item()
            self.scale_bias = scale_bias
        self.train()
        if use_cuda:
            self.cuda()
            Xtr = Xtr.cuda()
            ytr = ytr.cuda()
        def closure():
            optimizer.zero_grad()
            output = self(Xtr)
            loss = criterion(output, ytr)
            loss = loss + 0.5 * reg * self.weight.pow(2).sum()
            loss.backward()
            return loss

        for epoch in range(epochs):
            optimizer.step(closure)
        if self.bias is not None:
            self.bias.data.mul_(self.scale_bias)
        self.scale_bias = None

    def score(self, X, y):
        self.eval()
        with torch.no_grad():
            scores = self(X)
            scores = scores.argmax(-1)
            scores = scores.cpu()
        return torch.mean((scores == y).float()).item()
    


EPS = 1e-6


def normalize(x, p=2, dim=-1, inplace=True):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    if inplace:
        x.div_(norm.clamp(min=EPS))
    else:
        x = x / norm.clamp(min=EPS)
    return x

def spherical_kmeans(x, n_clusters, max_iters=100, block_size=None, verbose=True,
                     init=None, eps=1e-4):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x kmer_size x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    if x.ndim == 3:
        n_samples, kmer_size, n_features = x.size()
    else:
        n_samples, n_features = x.size()
    if init is None:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]

    prev_sim = np.inf
    tmp = x.new_empty(n_samples)
    assign = x.new_empty(n_samples, dtype=torch.long)
    if block_size is None or block_size == 0:
        block_size = x.shape[0]

    for n_iter in range(max_iters):
        for i in range(0, n_samples, block_size):
            end_i = min(i + block_size, n_samples)
            cos_sim = x[i: end_i].view(end_i - i, -1).mm(clusters.view(n_clusters, -1).t())
            tmp[i: end_i], assign[i: end_i] = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm(dim=-1, keepdim=True).clamp(min=EPS)

        if torch.abs(prev_sim - sim)/(torch.abs(sim)+1e-20) < 1e-6:
            break
        prev_sim = sim
    return clusters




def sinkhorn(dot, mask=None, eps=1e-03, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    if return_kernel:
        K = torch.exp(dot / eps)
    else:
        K = dot
    # K: n x in_size x out_size
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    a = float(out_size / in_size)
    if mask is not None:
        mask = mask.float()
        a = out_size / mask.sum(1, keepdim=True)
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        if mask is not None:
            u = u * mask
        v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
    if return_kernel:
        K = K / out_size
        return (K * dot).sum(dim=[1, 2])
    return K

def log_sinkhorn(K, mask=None, eps=1.0, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    batch_size, in_size, out_size = K.shape
    def min_eps(u, v, dim):
        Z = (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps
        return -torch.logsumexp(Z, dim=dim)
    # K: batch_size x in_size x out_size
    u = K.new_zeros((batch_size, in_size))
    v = K.new_zeros((batch_size, out_size))
    a = torch.ones_like(u).fill_(out_size / in_size)
    if mask is not None:
        a = out_size / mask.float().sum(1, keepdim=True)
    a = torch.log(a)
    for _ in range(max_iter):
        u = eps * (a + min_eps(u, v, dim=-1)) + u
        if mask is not None:
            u = u.masked_fill(~mask, -1e8)
        v = eps * min_eps(u, v, dim=1) + v
    if return_kernel:
        output = torch.exp(
            (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
        output = output / out_size
        return (output * K).sum(dim=[1, 2])
    K = torch.exp(
        (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
    return K

def multihead_attn(input, weight, mask=None, eps=1.0, return_kernel=False,
                   max_iter=100, log_domain=False, position_filter=None):
    """Comput the attention weight using Sinkhorn OT
    input: n x in_size x in_dim
    mask: n x in_size
    weight: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = input.shape
    m, out_size = weight.shape[:-1]
    K = torch.tensordot(input, weight, dims=[[-1], [-1]])
    K = K.permute(0, 2, 1, 3)
    if position_filter is not None:
        K = position_filter * K
    # K: n x m x in_size x out_size
    K = K.reshape(-1, in_size, out_size)
    # K: nm x in_size x out_size
    if mask is not None:
        mask = mask.repeat_interleave(m, dim=0)
    if log_domain:
        K = log_sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    else:
        if not return_kernel:
            K = torch.exp(K / eps)
        K = sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    # K: nm x in_size x out_size
    if return_kernel:
        return K.reshape(n, m)
    K = K.reshape(n, m, in_size, out_size)
    if position_filter is not None:
        K = position_filter * K
    K = K.permute(0, 3, 1, 2).contiguous()
    return K

def wasserstein_barycenter(x, c, eps=1.0, max_iter=100, sinkhorn_iter=50, log_domain=False):
    """
    x: n x in_size x in_dim
    c: out_size x in_dim
    """
    prev_c = c
    for i in range(max_iter):
        T = multihead_attn(x, c, eps=eps, log_domain=log_domain, max_iter=sinkhorn_iter)
        # T: n x out_size x in_size
        c = 0.5*c + 0.5*torch.bmm(T, x).mean(dim=0) / math.sqrt(c.shape[0])
        c /= c.norm(dim=-1, keepdim=True).clamp(min=1e-06)
        if ((c - prev_c) ** 2).sum() < 1e-06:
            break
        prev_c = c
    return c

def wasserstein_kmeans(x, n_clusters, out_size, eps=1.0, block_size=None, max_iter=100,
                       sinkhorn_iter=50, wb=False, verbose=True, log_domain=False, use_cuda=False):
    """
    x: n x in_size x in_dim
    output: n_clusters x out_size x in_dim
    out_size <= in_size
    """
    n, in_size, in_dim = x.shape
    if n_clusters == 1:
        if use_cuda:
            x = x.cuda()
        clusters = spherical_kmeans(x.view(-1, in_dim), out_size, block_size=block_size)
        if wb:
            clusters = wasserstein_barycenter(x, clusters, eps=0.1, log_domain=False)
        clusters = clusters.unsqueeze_(0)
        return clusters
    ## intialization
    indices = torch.randperm(n)[:n_clusters]
    clusters = x[indices, :out_size, :].clone()
    if use_cuda:
        clusters = clusters.cuda()

    wass_sim = x.new_empty(n)
    assign = x.new_empty(n, dtype=torch.long)
    if block_size is None or block_size == 0:
        block_size = n

    prev_sim = float('inf')
    for n_iter in range(max_iter):
        for i in range(0, n, block_size):
            end_i = min(i + block_size, n)
            x_batch = x[i: end_i]
            if use_cuda:
                x_batch = x_batch.cuda()
            tmp_sim = multihead_attn(x_batch, clusters, eps=eps, return_kernel=True, max_iter=sinkhorn_iter, log_domain=log_domain)
            tmp_sim = tmp_sim.cpu()
            wass_sim[i : end_i], assign[i: end_i] = tmp_sim.max(dim=-1)
        del x_batch
        sim = wass_sim.mean()
        if verbose and (n_iter + 1) % 10 == 0:
            print("Wasserstein spherical kmeans iter {}, objective value {}".format(
                  n_iter + 1, sim))

        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                idx = wass_sim.argmin()
                clusters[j].copy_(x[idx, :out_size, :])
                wass_sim[idx] = 1
            else:
                xj = x[index]
                if use_cuda:
                    xj = xj.cuda()
                c = spherical_kmeans(xj.view(-1, in_dim), out_size, block_size=block_size, verbose=False)
                if wb:
                    c = wasserstein_barycenter(xj, c, eps=0.001, log_domain=True, sinkhorn_iter=50)
                clusters[j] = c
        if torch.abs(prev_sim - sim) / sim.clamp(min=1e-10) < 1e-6:
            break
        prev_sim = sim
    return clusters