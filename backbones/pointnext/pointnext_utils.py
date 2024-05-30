import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from easydict import EasyDict as edict
import copy

import pointnet2_batch_cuda as pointnet2_cuda

CHANNEL_MAP = {
    'fj': lambda x: x,
    'df': lambda x: x,
    'assa': lambda x: x * 3,
    'assa_dp': lambda x: x * 3 + 3,
    'dp_fj': lambda x: 3 + x,
    'pj': lambda x: x,
    'dp': lambda x: 3,
    'pi_dp': lambda x: x + 3,
    'pj_dp': lambda x: x + 3,
    'dp_fj_df': lambda x: x*2 + 3,
    'dp_fi_df': lambda x: x*2 + 3,
    'pi_dp_fj_df': lambda x: x*2 + 6,
    'pj_dp_fj_df': lambda x: x*2 + 6,
    'pj_dp_df': lambda x: x + 6,
    'dp_df': lambda x: x + 3,
}

_ACT_LAYER = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    leakyrelu=nn.LeakyReLU,
    elu=nn.ELU,
    prelu=nn.PReLU,
    celu=nn.CELU,
    selu=nn.SELU,
    gelu=nn.GELU,
    sigmoid=nn.Sigmoid,
    tanh=nn.Tanh,
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
)


def create_act(act_args):
    """Build activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    if act_args is None:
        return None
    act_args = copy.deepcopy(act_args)
    
    if isinstance(act_args , str):
        act_args = {"act": act_args}    
    
    act = act_args.pop('act', None)
    if act is None:
        return None

    if isinstance(act, str):
        act = act.lower()
        assert act in _ACT_LAYER.keys(), f"input {act} is not supported"
        act_layer = _ACT_LAYER[act]

    inplace = act_args.pop('inplace', True)

    if act not in ['gelu', 'sigmoid']: # TODO: add others
        return act_layer(inplace=inplace, **act_args)
    else:
        return act_layer(**act_args)
    

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial BCHW tensors """

    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()


class LayerNorm1d(nn.LayerNorm):
    """ LayerNorm for channels of '1D' spatial BCN tensors """

    def __init__(self, num_channels, **kwargs):
        super().__init__(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 2, 1).contiguous()


class FastBatchNorm1d(nn.Module):
    """Fast BachNorm1d for input with shape [B, N, C], where the feature dimension is at last. 
    Borrowed from torch-points3d: https://github.com/torch-points3d/torch-points3d
    """
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, **kwargs)

    def _forward_dense(self, x):
        return self.bn(x.transpose(1,2)).transpose(2, 1)

    def _forward_sparse(self, x):
        return self.bn(x)

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))


_NORM_LAYER = dict(
    bn1d=nn.BatchNorm1d,
    bn2d=nn.BatchNorm2d,
    bn=nn.BatchNorm2d,
    in2d=nn.InstanceNorm2d, 
    in1d=nn.InstanceNorm1d, 
    gn=nn.GroupNorm,
    syncbn=nn.SyncBatchNorm,
    ln=nn.LayerNorm,    # for tokens
    ln1d=LayerNorm1d,   # for point cloud
    ln2d=LayerNorm2d,   # for point cloud
    fastbn1d=FastBatchNorm1d, 
    fastbn2d=FastBatchNorm1d, 
    fastbn=FastBatchNorm1d, 
)


def create_norm(norm_args, channels, dimension=None):
    """Build normalization layer.
    Returns:
        nn.Module: Created normalization layer.
    """
    if norm_args is None:
        return None
    if isinstance(norm_args, dict):    
        norm_args = edict(copy.deepcopy(norm_args))
        norm = norm_args.pop('norm', None)
    else:
        norm = norm_args
        norm_args = edict()
    if norm is None:
        return None
    if isinstance(norm, str):
        norm = norm.lower()
        if dimension is not None:
            dimension = str(dimension).lower()
            if dimension not in norm:
                norm += dimension
        assert norm in _NORM_LAYER.keys(), f"input {norm} is not supported"
        norm = _NORM_LAYER[norm]
    return norm(channels, **norm_args)


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv2d, self).__init__(*args, (1, 1), **kwargs)
        else:
            super(Conv2d, self).__init__(*args, **kwargs)


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        if len(args) == 2 and 'kernel_size' not in kwargs.keys():
            super(Conv1d, self).__init__(*args, 1, **kwargs)
        else:
            super(Conv1d, self).__init__(*args, **kwargs)


def create_convblock2d(*args,
                       norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)
    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)

    elif order == 'norm-act-conv':
        conv_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        conv_layer.append(Conv2d(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='2d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv2d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        if norm_layer is not None:
            conv_layer.append(norm_layer)
    else:
        raise NotImplementedError(f"{order} is not supported")

    return nn.Sequential(*conv_layer)


def create_convblock1d(*args,
                       norm_args=None, act_args=None, order='conv-norm-act', **kwargs):
    out_channels = args[1]
    in_channels = args[0]
    bias = kwargs.pop('bias', True)
    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)

    elif order == 'norm-act-conv':
        conv_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            conv_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        conv_layer.append(Conv1d(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        conv_layer = [Conv1d(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            conv_layer.append(act_layer)
        if norm_layer is not None:
            conv_layer.append(norm_layer)
    else:
        raise NotImplementedError(f"{order} is not supported")

    return nn.Sequential(*conv_layer)


def random_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint), device=xyz.device)
    return idx


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
        # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2_cuda.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


def create_linearblock(*args,
                       norm_args=None,
                       act_args=None,
                       order='conv-norm-act',
                       **kwargs):
    in_channels = args[0]
    out_channels = args[1]
    bias = kwargs.pop('bias', True)

    if order == 'conv-norm-act':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        linear_layer = [nn.Linear(*args, bias, **kwargs)]
        if norm_layer is not None:
            linear_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
    elif order == 'norm-act-conv':
        linear_layer = []
        norm_layer = create_norm(norm_args, in_channels, dimension='1d')
        bias = kwargs.pop('bias', True)
        bias = False if norm_layer is not None else bias
        if norm_layer is not None:
            linear_layer.append(norm_layer)
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
        linear_layer.append(nn.Linear(*args, bias=bias, **kwargs))

    elif order == 'conv-act-norm':
        norm_layer = create_norm(norm_args, out_channels, dimension='1d')
        bias = False if norm_layer is not None else bias
        linear_layer = [nn.Linear(*args, bias=bias, **kwargs)]
        act_layer = create_act(act_args)
        if act_args is not None:
            linear_layer.append(act_layer)
        if norm_layer is not None:
            linear_layer.append(norm_layer)

    return nn.Sequential(*linear_layer)