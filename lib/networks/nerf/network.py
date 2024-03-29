import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.networks.nerf.nerf import NeRF
from lib.config import cfg
from lib.utils import rays_utils
import ipdb

DEBUG = False

# 属性： xyz_encoder + dir_encoder + mlp（linear + relu)  
# 方法： render  batchify  forward
class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        # task arg
        self.task_arg = cfg.task_arg

        self.coarse_NeRF = NeRF().cuda()
        # construct fine NeRF
        if (len(self.task_arg.cascade_samples) > 1):
            self.fine_NeRF = NeRF().cuda()

        self.use_viewdirs = cfg.network.nerf.use_viewdirs


    def render(self, rays):
        # uv_encoding = self.uv_encoder(uv)   # [8192, 42]
        # x = uv_encoding
        # for i, l in enumerate(self.backbone_layer):
        #     x = self.backbone_layer[i](x)
        #     x = F.relu(x)
        # rgb = self.output_layer(x)
        # return {'rgb': rgb}         # [8192, 3]
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:,0:3], rays[:,3:6] # [N_rays, 3] each
        viewdirs = rays[:,-3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [N_rays, 1]

        t_vals = torch.linspace(0., 1., self.task_arg.cascade_samples[0])
        t_vals = t_vals.to(rays.device)
        z_vals = near * (1. - t_vals) + far * (t_vals)          # [N_rays, samples]

        mids = .5 * (z_vals[..., 1:]) + .5 * (z_vals[..., :-1]) # [N_rays, samples - 1]
        upper = torch.cat([mids, z_vals[..., -1:]], -1)         # [N_rays, samples]
        lower = torch.cat([mids, z_vals[..., :1]], -1)          # [N_rays, samples]
        # stratified sample
        t_rand = torch.rand(z_vals.shape).to(rays.device)

        z_vals = lower + (upper - lower) * t_rand               # [1024, 64]

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]   #  [1024, 64, 3]
        viewdirs = viewdirs[:, None].expand(pts.shape)

        raw = self.coarse_NeRF(pts, viewdirs)

        rgb_map, disp_map, acc_map, weights, depth_map = rays_utils.raw2outputs(raw, z_vals, rays_d, self.task_arg.white_bkgd)
        
        if len(self.task_arg.cascade_samples) > 1:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = rays_utils.sample_pdf(z_vals_mid, weights[..., 1:-1], self.task_arg.cascade_samples[1])

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
            
            viewdirs = torch.repeat_interleave(viewdirs, repeats=3, dim=1)

            # fine NeRF output
            raw = self.fine_NeRF(pts, viewdirs)
            rgb_map, disp_map, acc_map, weights, depth_map = rays_utils.raw2outputs(raw, z_vals, rays_d, self.task_arg.white_bkgd)
            
        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
        
        if (len(self.task_arg.cascade_samples) > 1):
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret
        


    def batchify(self, rays):
        # all_ret = {}
        # chunk = cfg.task_arg.chunk_size
        # for i in range(0, uv.shape[0], chunk):
        #     ret = self.render(uv[i:i + chunk], batch)
        #     for k in ret:
        #         if k not in all_ret:
        #             all_ret[k] = []
        #         all_ret[k].append(ret[k])
        # all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        # return all_ret
        all_ret = {}
        for i in range(0, rays.shape[0], self.task_arg.chunk_size):
            ret = self.render(rays[i:i+self.task_arg.chunk_size])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                    all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret



    # 直接求出 predicted rgb
    def forward(self, batch):
        # batch: rays_o + rays_d + rgb
        # sampling points
        rays_o = batch['rays_o'].reshape(-1, 3)    
        rays_d = batch['rays_d'].reshape(-1, 3)
        N_rays = rays_o.shape[0]
        viewdirs = F.normalize(rays_d, dim=1)
        near, far = 2. * torch.ones([N_rays, 1]), 6. * torch.ones([N_rays, 1])

        near, far = near.to(rays_o.device), far.to(rays_o.device)

        rays = torch.cat([rays_o, rays_d, near, far], -1)   # [N_rays, 8]
        if self.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)          # [N_rays, 11]

        all_ret = self.batchify(rays)

        return all_ret