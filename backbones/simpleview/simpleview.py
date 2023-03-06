# Source: https://github.com/princeton-vl/SimpleView/blob/master/models/mv.py

import torch
from torch import nn

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from resnet import get_resnet, BasicBlock, Squeeze
from simpleview_utils import PCViews


class MVModel(nn.Module):
    def __init__(self, feat_size=16):

        super().__init__()
        self.feat_size = feat_size

        pc_views = PCViews()
        self.num_views = pc_views.num_views
        self._get_img = pc_views.get_img

        self.backbone, self.out_dim = self.get_backbone(feat_size)

    def forward(self, x):
        """
        :param pc:
        :return:
        """
        img = self.get_img(x)
        out = self.backbone(img)
        return out.view(-1, self.num_views, self.out_dim)

    def get_img(self, pc):
        img = self._get_img(pc).float()
        img = img.to(next(self.parameters()).device)
        assert len(img.shape) == 3
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)

        return img

    @staticmethod
    def get_backbone(feat_size):
        """
        Return layers for the image model
        """
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = get_resnet(
            block=block,
            layers=layers,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        backbone = nn.Sequential(*img_layers)

        return backbone, in_features