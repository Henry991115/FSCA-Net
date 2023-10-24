# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet.py
# @Software: PyCharm
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans_epa import ChannelTransformer


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        #print("down",out.size())
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out



class DPCABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(DPCABlock, self).__init__()

        # 设计自适应卷积核
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.conv1(avg_pool_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.conv2(avg_pool_g.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out


class UpBlock_attention_high(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.coatt_1 = DPCABlock(channels=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        #skip_x_att = self.coatt(g=up, x=skip_x)
        skip_x_att = self.coatt_1(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UpBlock_attention_low(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.coatt_1 = DPCABlock(channels=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        #skip_x_att = self.coatt(g=up, x=skip_x)
        skip_x_att = self.coatt_1(g=x, x=skip_x)
        x = torch.cat([skip_x_att, x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class bridge(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        #self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.coatt_1 = DPCABlock(channels=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        #x = self.up(x)
        #print(up.size())
        x = _upsample_like(x, skip_x)
        skip_x_att = self.coatt_1(g=x, x=skip_x)
        x = torch.cat([skip_x_att, x], dim=1)
        #print(x.size())
        return self.nConvs(x)


class FSCANet(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.bridge = bridge(in_channels*16,in_channels*8, nb_Conv=2)
        self.up4 = UpBlock_attention_low(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention_high(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention_high(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention_high(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        print([x1.size(), x2.size(), x3.size(), x4.size()])
        x5 = self.down4(x4)
        x4_1 = x4
        # print(x4_1.size())
        # print(x5.size())
        # x1,x2,x3,x4,att_weights = self.mtc(x1,x2,x3,x4)
        x1, x2, x3, x4 = self.mtc(x1, x2, x3, x4)
        print([x1.size(), x2.size(), x3.size(), x4.size()])
        x = self.bridge(x5, x4_1)
        # x5_1 = _upsample_like(x5, x4)
        # x5_1 = nn.Upsample(x5, scale_factor=2)
        # print(x5_1.size)
        # x = torch.cat((x4_1, x5_1), dim=1)
        # print(x.size())
        # print(x4.size())
        x = self.up4(x, x4)
        print(x.size())
        x = self.up3(x, x3)
        print(x.size())
        x = self.up2(x, x2)
        print(x.size())
        x = self.up1(x, x1)
        print(x.size())
        if self.n_classes ==1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x) # if not using BCEWithLogitsLoss or class>1
        # if self.vis: # visualize the attention maps
        #     return logits, att_weights
        # else:
        #     return logits
        return  logits




