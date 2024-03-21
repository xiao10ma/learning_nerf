import torch
import torch.nn as nn
from lib.utils import net_utils
from lib.config import cfg

# 优化器，mse+psnr
class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

    def forward(self, batch):
        output = self.net(batch)

        scalar_stats = {}
        loss = 0
        color_loss = self.color_crit(output['rgb'], batch['rgb'])
        scalar_stats.update({'color_mse': color_loss})
        loss += color_loss

        # psnr一般只用与评估
        # color_loss.detach()了，返回一个新的张量，但是脱离了计算图
        # 所以color_loss本身还是会被反向传播更新的，但是其新生成的副本不会参与反向传播，从而psnr也不会参与到反向传播
        psnr = -10. * torch.log(color_loss.detach()) / \
                torch.log(torch.Tensor([10.]).to(color_loss.device))    # device = cuda0
        scalar_stats.update({'psnr': psnr})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
