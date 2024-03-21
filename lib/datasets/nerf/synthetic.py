import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2

def sample_rays_np(H, W, f, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5+.5)/f, -(j-H*.5+.5)/f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

# 1024 条ray <==> 1024个GT rgb
# ray 通过 内参矩阵 + 外参矩阵求
class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        cams = kwargs['cams']
        start, end, step = cams
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays
        self.frame_cnt = 0

        # read image, poses(transfom matrix)
        self.imgs = []
        self.poses = []

        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))

        if end == -1:
            end = len(json_info['frames'])

        for i in range(start, end, step):
            frame = json_info['frames'][i]
            img_path = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            img = imageio.imread(img_path)
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            self.imgs.append(img)
            self.poses.append(np.array(frame['transform_matrix']))
            self.frame_cnt += 1

        self.imgs = (np.array(self.imgs) / 255.).astype(np.float32)
        self.poses = np.array(self.poses).astype(np.float32)

        self.H, self.W = self.imgs[0].shape[:2]
        camera_angle_x = float(json_info['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)

        if self.input_ratio != 1.:     
            self.H = self.H // 2
            self.W = self.W // 2
            self.focal = self.focal / 2.

            imgs_half_res = np.zeros((self.imgs.shape[0], self.H, self.W, 3))
            for i, img in enumerate(self.imgs):
                imgs_half_res[i] = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
            self.imgs = imgs_half_res

    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(self.H * self.W, self.batch_size, replace=False)
            # gt rgb
            rgb = self.imgs[index].reshape(-1, 3)[ids]
            # sample points  
            rays_o, rays_d = sample_rays_np(self.H, self.W, self.focal, np.linalg.inv(self.poses[index]))   
        else:
            rgb = self.imgs[index].reshape(-1, 3)
            rays_o, rays_d = sample_rays_np(self.H, self.W, self.focal, np.linalg.inv(self.poses[index]))   

        ret = {'rays_o' : rays_o, 'rays_d' : rays_d, 'rgb' : rgb}
        ret.update({'meta' : {'H' : self.H, 'W' : self.W}})
        return ret


    def __len__(self):
        return self.frame_cnt