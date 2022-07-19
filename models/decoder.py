from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb

from models.transformers.local_attention import AttentionLayer3D


class sepConv3dBlock(nn.Module):
    '''
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1)):
        super(sepConv3dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes,stride)
        self.conv1 = sepConv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = sepConv3d(out_planes, out_planes, 3, (1,1,1), 1)
            

    def forward(self,x):
        out = F.relu(self.conv1(x), inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out), inplace=True)
        return out

class projfeat3d(nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1,1), padding=(0,0), stride=stride[:2],bias=False)
        self.bn = nn.GroupNorm(8, out_planes)

    def forward(self,x):
        b,c,d,h,w = x.size()
        x = self.conv1(x.view(b,c,d,h*w))
        x = self.bn(x)
        x = x.view(b,-1,d//self.stride[0],h,w)
        return x

def sepConv3d(in_planes, out_planes, kernel_size, stride, pad, bias=False): #Original: False -> uses BatchNorm
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                         nn.GroupNorm(out_planes//4, out_planes))

class decoderBlock(nn.Module):
    def __init__(self, nconvs, inchannelF, channelF, stride=(1,1,1), nstride=1, att=False, heads=1):
        super(decoderBlock, self).__init__()
        self.att=att
        stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)
        self.convs = [sepConv3dBlock(inchannelF,channelF,stride=stride[0])]
        for i in range(1,nconvs):
            self.convs.append(sepConv3dBlock(channelF,channelF, stride=stride[i]))
        self.convs = nn.Sequential(*self.convs)

        self.classify = nn.Sequential(sepConv3d(channelF, channelF, 3, (1,1,1), 1),
                                       nn.ReLU(inplace=True),
                                       sepConv3d(channelF, 1, 3, (1,1,1),1,bias=True))

        if att:
            self.attention = nn.Sequential(nn.Conv3d(channelF, channelF, kernel_size=3, padding=1, stride=1, bias=False),
                nn.GroupNorm(channelF//4, channelF),
                nn.ReLU(inplace=True),
                AttentionLayer3D(channelF, channelF, 3, 1, 1, heads, bias=False))

            self.diag_weights = torch.FloatTensor(channelF)
            self.diag_weights = self.diag_weights.fill_(0.1)
            self.diag_matrix = nn.Parameter(torch.diag(self.diag_weights), requires_grad=True)
 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()


    def forward(self,fvl):

        fvl = self.convs(fvl)
        # attention
        if self.att:
            res = fvl
            out = self.attention(fvl)
            # print("shapes: ", out.shape, self.diag_matrix.shape)
            out = torch.einsum('b c d h w, c c -> b c d h w', out, self.diag_matrix)
            out += res
        else:
            out = fvl

        return self.classify(out).squeeze(1)
