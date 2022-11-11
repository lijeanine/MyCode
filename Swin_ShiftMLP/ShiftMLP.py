import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
#from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
#from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
#import pdb

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):#stage 4:H=16,W=16
        # pdb.set_trace()
        B, N, C = x.shape #stage 4: x.shape=[4,16*16,160]
        #shifted-MLP across width
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous() #xn.shape=[4,160,16,16]
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0) #pad 2 elements for each of the last two dimensions,xn.shape=[4,160,20,20]
        xs = torch.chunk(xn, self.shift_size, 1)#xs is tuple,len(xs)=5,xs[i].shape=[4,32,20,20]
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))] #window 16 of width 
            #x_shift is list,len(x_shift)=5,x_shift[i].shape=[4,32,20,20]
            #for i in [-2,-1,0,-1,2]:shift = i 
            #for i in (0,5):x_c.shape=[4,32,20,20] 
        x_cat = torch.cat(x_shift, 1) #x_cat.shape=[4,160,20,20]
        x_cat = torch.narrow(x_cat, 2, self.pad, H)#x_cat.shape=[4,160,16,20] slices [2,17] in dim 2 
        x_s = torch.narrow(x_cat, 3, self.pad, W) #x_s.shape=[4,160,16,16] slice [2,17] in dim 3


        x_s = x_s.reshape(B,C,H*W).contiguous() #x_s.shape=[4,160,16*16]
        x_shift_r = x_s.transpose(1,2) #x_shift_r.shape=[4,16*16,160]


        x = self.fc1(x_shift_r) #x.shape=[4,256,160]

        x = self.dwconv(x, H, W) #out:x.shape=[4,256,160]
        x = self.act(x) 
        x = self.drop(x)#out:x.shape=[4,256,160]

        #shifted-MLP across height which converts the dimensions from H to O 
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous() #xn.shape=[4,160,16,16]
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)#xn.shape=[4,160,20,20]
        xs = torch.chunk(xn, self.shift_size, 1)#len(xs)=5,xs[i].shape=[4,32,20,20]
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)#x_shift_c.shape=[4,16*16,160]

        x = self.fc2(x_shift_c)#x.shape=[4,256,160]
        x = self.drop(x)#x.shape=[4,256,160]
        return x



if __name__=='__main__':
    self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
            # mlp_hidden_dim = int(dim * mlp_ratio) #H ,W maybe matched with input_resolution
    self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim,  drop=drop)
