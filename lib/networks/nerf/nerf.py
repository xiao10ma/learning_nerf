import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg
from lib.utils import rays_utils
import ipdb

# 属性： xyz_encoder + dir_encoder + mlp（linear + relu) 
# init：构建神经网络
# forward：输入pts + viewdirs，返回raw（rgb + alpha） 
class NeRF(nn.Module):
    def __init__(self,):
        super(NeRF, self).__init__()
        net_cfg = cfg.network
        # get encoder
        self.xyz_encoder, self.xyz_input_ch = get_encoder(net_cfg.xyz_encoder)   # xyz_encoder 63
        self.dir_encoder, self.dir_input_ch = get_encoder(net_cfg.dir_encoder)   # dir_encoder 27

        D, W  = net_cfg.nerf.D, net_cfg.nerf.W
        self.skips = net_cfg.nerf.skips
        self.use_viewdirs = net_cfg.nerf.use_viewdirs

        self.pts_linear = nn.ModuleList(
            [nn.Linear(self.xyz_input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.xyz_input_ch, W) for i in range(D - 1)]
        )

        self.views_linear = nn.ModuleList([nn.Linear(self.dir_input_ch + W, W//2)])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, 4)

    def forward(self, pts, views=None):
        encoded_xyz = self.xyz_encoder(pts)
        if (views is not None):
            encoded_dir = self.dir_encoder(views)
        
        h = encoded_xyz
        for i, l in enumerate(self.pts_linear):
            h = self.pts_linear[i](h)
            h = F.relu(h)
            # resnet
            if i in self.skips:
                h = torch.cat([encoded_xyz, h], -1)
            
        if (self.use_viewdirs):
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, encoded_dir], -1)
        
            for i, l in enumerate(self.views_linear):
                h = self.views_linear[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs    