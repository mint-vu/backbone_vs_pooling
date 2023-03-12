import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1])) # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2])) # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st-org_x
    edge2 = neighbor_2nd-org_x
    normals = torch.cross(edge1, edge2, dim=1) # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True) # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True) # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1) # B,14,N

    return new_pts

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature
class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""
    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(1024//8)
        self.bn2 = nn.BatchNorm1d(1024//8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat) 
        proj_key = self.key_conv(x_hat).permute(0, 2, 1) 
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat)-similarity_mat
        affinity_mat = self.softmax(affinity_mat)
        
        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha*out + x 
        return out
    
    
class ABEM_Module(nn.Module):
    """ Attentional Back-projection Edge Features Module (ABEM)"""
    def __init__(self, in_dim, out_dim, k):
        super(ABEM_Module, self).__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Sequential(nn.Conv2d(out_dim, in_dim, kernel_size=[1,self.k], bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa1 = CAA_Module(out_dim)

        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=[1,self.k], bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa2 = CAA_Module(out_dim)

    def forward(self,x):
        # Prominent Feature Encoding
        x1 = x # input
        input_edge = get_graph_feature(x, k=self.k)
        x = self.conv1(input_edge)
        x2 = x # EdgeConv for input

        x = self.conv2(x) # LFC
        x = torch.squeeze(x, -1)
        x3 = x # Back-projection signal

        delta = x3 - x1 # Error signal

        x = get_graph_feature(delta, k=self.k)  # EdgeConv for Error signal
        x = self.conv3(x)
        x4 = x

        x = x2 + x4 # Attentional feedback
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.caa1(x) # B,out_dim,N

        # Fine-grained Feature Encoding
        x_local = self.conv4(input_edge)
        x_local = torch.squeeze(x_local, -1) 
        x_local = self.caa2(x_local) # B,out_dim,N

        return x, x_local
    
class GBNet(nn.Module):
    def __init__(self):
        super(GBNet, self).__init__()
        
        self.k = 20

        self.abem1 = ABEM_Module(14, 64, self.k)
        self.abem2 = ABEM_Module(64, 64, self.k)
        self.abem3 = ABEM_Module(64, 128, self.k)
        self.abem4 = ABEM_Module(128, 256, self.k)

        self.bn = nn.BatchNorm1d(1024)
        self.conv = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                self.bn,
                                nn.LeakyReLU(negative_slope=0.2))
    
        self.caa = CAA_Module(1024)


    def forward(self, x):
        # x: B,3,N
        batch_size = x.size(0)
        B,N,D = x.shape
        x=x.view(B,D,N)

        # Geometric Point Descriptor:
        x = geometric_point_descriptor(x) # B,14,N

        # 1st Attentional Back-projection Edge Features Module (ABEM):
        x1, x1_local = self.abem1(x)

        # 2nd Attentional Back-projection Edge Features Module (ABEM):
        x2, x2_local = self.abem2(x1)

        # 3rd Attentional Back-projection Edge Features Module (ABEM):
        x3, x3_local = self.abem3(x2)

        # 4th Attentional Back-projection Edge Features Module (ABEM):
        x4, x4_local = self.abem4(x3)

        # Concatenate both prominent and fine-grained outputs of 4 ABEMs:
        x = torch.cat((x1, x1_local, x2, x2_local, x3, x3_local, x4, x4_local), dim=1)  # B,(64+64+128+256)x2,N
        x = self.conv(x) 
        x = self.caa(x) # B,1024,N

        return x
