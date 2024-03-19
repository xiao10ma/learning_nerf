import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio
import json
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        view = kwargs['view']
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_pixels         # 8192

        # read image
        image_paths = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format('train'))))
        for frame in json_info['frames']:
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))

        img = imageio.imread(image_paths[view])/255.                # 读 view 号图片
        img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])    # alpha混合，背景为白色
        if self.input_ratio != 1.:                                  # 根据 input_ratio 缩放图片
            img = cv2.resize(img, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
        # set image
        self.img = np.array(img).astype(np.float32)         # img [0 - 1]
        # set uv
        H, W = img.shape[:2]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))      
        u, v = X.astype(np.float32) / (W-1), Y.astype(np.float32) / (H-1)   # u, v [0 - 1]
        self.uv = np.stack([u, v], -1).reshape(-1, 2).astype(np.float32)    # [800 * 800, 2]

    # 返回字典 {'uv', 'rgb', 'meta'}
    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(len(self.uv), self.batch_size, replace=False)        # 从 [0, len(self.uv)) 中取出batch_size个sample，replace表示放回：True就代表可以取相同数字，False代表不能拿相同的
            uv = self.uv[ids]
            rgb = self.img.reshape(-1, 3)[ids]
        else:
            uv = self.uv
            rgb = self.img.reshape(-1, 3)
        ret = {'uv': uv, 'rgb': rgb} # input and output. they will be sent to cuda
        ret.update({'meta': {'H': self.img.shape[0], 'W': self.img.shape[1]}}) # meta means no need to send to cuda
        return ret

    def __len__(self):
        # we only fit 1 images, so we return 1
        return 1
