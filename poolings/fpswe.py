# Source: https://github.com/navid-naderi/PSWE/blob/main/pswe.py

import torch
import torch.nn as nn


def interp1d(x,y,xnew,device):
    
    M,N=xnew.shape
    ynew = torch.zeros((M,N), device=device)
    ind = ynew.long()
    
    if N==1:
        torch.searchsorted(x.contiguous().squeeze(),
                               xnew.contiguous(), out=ind)
    else:
        torch.searchsorted(x.contiguous().squeeze(),
                               xnew.contiguous().squeeze(), out=ind)
    eps = torch.finfo(y.dtype).eps
    slopes = (y[:, 1:]-y[:, :-1])/(eps + (x[:, 1:]-x[:, :-1]))
    ind -= 1
    ind = torch.clamp(ind, 0, x.shape[1] - 1 - 1)
    
    def sel(x):

        return torch.gather(x, 1, ind)
    
    ynew = sel(y) + sel(slopes)*(xnew-sel(x))
    
    return ynew
    

class FPSWE(nn.Module):
    def __init__(self, d_in, num_ref_points=1024, num_projections=1024):
        '''
        The PSWE module that produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size.
        Inputs:
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_projections: Number of slices
        '''
        super(FPSWE, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_projections = num_projections

        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, num_projections)
        # self.reference = uniform_ref.to(device)
        self.reference = nn.Parameter(uniform_ref, requires_grad=False)

        # slicer
        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_projections, bias=False), dim=0)
        self.theta.requires_grad=False
        if num_projections <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data, requires_grad=False)
        self.theta.weight_g.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Parameter(torch.zeros(num_projections, num_ref_points))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        Output:
            weighted_embeddings: B x num_projections tensor, containing a batch of B embeddings, each of dimension "num_projections" (i.e., number of slices)
        '''

        B, N, dn = X.shape
        Xslices = self.get_slice(X)
        #print(Xslices.shape)
        #L = Xslices.shape[2]
        Xslices_sorted, Xind = torch.sort(Xslices, dim=1)


        M, dm = self.reference.shape

        if M == N:
            Xslices_sorted_interpolated = Xslices_sorted
        else:
            x = torch.linspace(0, 1, N + 2)[1:-1].unsqueeze(0).repeat(B * self.num_projections, 1).to(X.device)
            xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_projections, 1).to(X.device)
            y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_projections, -1)
            Xslices_sorted_interpolated = torch.transpose(interp1d(x, y, xnew,device=X.device).view(B, self.num_projections, -1), 1, 2)
            
        
        #print(Xslices_sorted_interpolated.shape)
            
            
            #print('y shape is',y.shape)
            #Xslices_sorted_interpolated = torch.transpose(Interp1d().apply(x, y, xnew).view(B, self.num_projections, -1), 1, 2)
            #print('Xslices_sorted_interpolated shape is',Xslices_sorted_interpolated.shape)

        Rslices = self.reference.expand(Xslices_sorted_interpolated.shape)

        _, Rind = torch.sort(Rslices, dim=1)
        embeddings = (Rslices - torch.gather(Xslices_sorted_interpolated, dim=1, index=Rind)).permute(0, 2, 1)

        w = self.weight.unsqueeze(0).repeat(B, 1, 1)
        weighted_embeddings = (w * embeddings).sum(-1)
        #print('weighted_embeddings shape is',weighted_embeddings.shape)
        return weighted_embeddings


    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)
