#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(lxtpku@pku.edu.cn)
# Pytorch Implementation of Octave Conv Operation
# This version use nn.Conv2d because alpha_in always equals alpha_out

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import BACKBONES, build_backbone


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.h2g_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv3d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        # self.l2h = torch.nn.Conv3d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
        #                            kernel_size, 1, padding, dilation, groups, bias)
        # self.h2l = torch.nn.Conv3d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
        #                            kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv3d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    # def forward(self, x):
    #     X_h, X_l = x

    #     if self.stride ==2:
    #         X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

    #     X_h2l = self.h2g_pool(X_h)

    #     X_h2h = self.h2h(X_h)
    #     X_l2h = self.l2h(X_l)

    #     X_l2l = self.l2l(X_l)
    #     X_h2l = self.h2l(X_h2l)
        
    #     X_l2h = self.upsample(X_l2h)
    #     X_h = X_l2h + X_h2h
    #     X_l = X_h2l + X_l2l

    #     return X_h, X_l

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        return self.h2h(X_h), self.l2l(X_l)




class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        self.h2g_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.h2l = torch.nn.Conv3d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv3d(in_channels, out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    # def forward(self, x):
    #     if self.stride ==2:
    #         x = self.h2g_pool(x)

    #     X_h2l = self.h2g_pool(x)
    #     X_h = x
    #     X_h = self.h2h(X_h)
    #     X_l = self.h2l(X_h2l)

    #     return X_h, X_l
    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        # X_h2l = self.h2g_pool(x)
        # X_h = x
        # X_h = self.h2h(X_h)
        # X_l = self.h2l(X_h2l)

        return self.h2h(x), self.h2l(self.h2g_pool(x))


class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        self.h2g_pool = nn.AvgPool3d(kernel_size=2, stride=2)

        self.l2h = torch.nn.Conv3d(int(alpha * in_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv3d(in_channels - int(alpha * in_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    # def forward(self, x):
    #     X_h, X_l = x

    #     if self.stride ==2:
    #         X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

    #     X_l2h = self.l2h(X_l)
    #     X_h2h = self.h2h(X_h)
    #     X_l2h = self.upsample(X_l2h)
        
    #     X_h = X_h2h + X_l2h

    #     return X_h
    
    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        # X_l2h = self.l2h(X_l)
        # X_h2h = self.h2h(X_h)
        # X_l2h = self.upsample(X_l2h)
        
        X_h = self.h2h(X_h) + self.upsample(self.l2h(X_l))

        return X_h


class OctaveCBR(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3,alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCBR, self).__init__()
        self.conv = OctaveConv(in_channels,out_channels,kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(int(out_channels*(1-alpha)))
        self.bn_l = norm_layer(int(out_channels*alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class OctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveCB, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class FirstOctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False,norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCBR, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        x_l = self.relu(self.bn_l(x_l))
        return x_h, x_l


class LastOCtaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCBR, self).__init__()
        self.conv = LastOctaveConv(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.relu(self.bn_h(x_h))
        return x_h


class FirstOctaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha=0.5,stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(FirstOctaveCB, self).__init__()
        self.conv = FirstOctaveConv(in_channels,out_channels,kernel_size, alpha,stride,padding,dilation,groups,bias)
        self.bn_h = norm_layer(int(out_channels * (1 - alpha)))
        self.bn_l = norm_layer(int(out_channels * alpha))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l)
        return x_h, x_l


class LastOCtaveCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(LastOCtaveCB, self).__init__()
        self.conv = LastOctaveConv( in_channels, out_channels, kernel_size, alpha, stride, padding, dilation, groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv(x)
        x_h = self.bn_h(x_h)
        return x_h


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None,First=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.first = First
        if self.first:
            self.ocb1 = FirstOctaveCBR(inplanes, inplanes, kernel_size=3,norm_layer=norm_layer,padding=1)
            self.ocb1_1 = OctaveCBR(inplanes, width, kernel_size=3,norm_layer=norm_layer,padding=1)
        else:
            self.ocb1 = OctaveCBR(inplanes, width, kernel_size=3,norm_layer=norm_layer,padding=1)

        self.ocb2 = OctaveCBR(width, width, kernel_size=3, stride=stride, groups=groups, norm_layer=norm_layer)
        self.ocb3 = OctaveCB(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        if self.first:
            x_h_res, x_l_res = self.ocb1(x)
            x_h, x_l = self.ocb1_1((x_h_res,x_l_res))
            x_h, x_l = self.ocb2((x_h, x_l))
        else:
            x_h_res, x_l_res = x
            x_h, x_l = self.ocb1((x_h_res,x_l_res))
            x_h, x_l = self.ocb2((x_h, x_l))

        x_h, x_l = self.ocb3((x_h, x_l))

        if self.downsample is not None:
            x_h_res, x_l_res = self.downsample((x_h_res,x_l_res))

        x_h += x_h_res
        x_l += x_l_res

        x_h = self.relu(x_h)
        x_l = self.relu(x_l)

        return x_h, x_l

class BottleneckOrigin(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckOrigin, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 =  nn.Conv3d(inplanes, width, kernel_size=1, stride=stride, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, padding=1, bias=False, dilation=1)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckLast(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BottleneckLast, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        # Last means the end of two branch
        self.ocb1 = OctaveCBR(inplanes, width,kernel_size=1,padding=0)
        self.ocb2 = OctaveCBR(width, width, kernel_size=3, stride=stride, groups=groups, norm_layer=norm_layer, padding=1)
        self.ocb3 = LastOCtaveCB(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):

        x_h_res, x_l_res = x
        x_h, x_l = self.ocb1((x_h_res, x_l_res))

        x_h, x_l = self.ocb2((x_h, x_l))
        x_h = self.ocb3((x_h, x_l))

        if self.downsample is not None:
            x_h_res = self.downsample((x_h_res, x_l_res))

        x_h += x_h_res
        x_h = self.relu(x_h)

        return x_h

def make_layer(inplanes, planes, blocks, stride=1, norm_layer=None, First=False):

    downsample = None
    if (stride != 1) or (inplanes != planes * Bottleneck.expansion):
        downsample = nn.Sequential(
            OctaveCB(in_channels=inplanes,out_channels=planes * Bottleneck.expansion, kernel_size=1, stride=stride, padding=0)
        )

    layers = []
    layers.append(Bottleneck(inplanes, planes, stride, downsample, First=First))
    for _ in range(1, blocks):
        layers.append(Bottleneck(planes, planes))

    return nn.Sequential(*layers)


def make_last_layer(inplanes, planes, blocks, stride=1, norm_layer=None):

    # downsample = None
    # if stride != 1 or inplanes != planes * Bottleneck.expansion:
    #     downsample = nn.Sequential(
    #         LastOCtaveCB(in_channels=inplanes,out_channels=planes * Bottleneck.expansion, kernel_size=1, stride=stride, padding=0)
    #     )

    # layers = []
    # layers.append(BottleneckLast(inplanes, planes, stride, downsample))

    # for _ in range(1, blocks):
    #     layers.append(BottleneckOrigin(planes, planes))

    downsample = None
    if stride != 1 or inplanes != planes * Bottleneck.expansion:
        downsample = nn.Sequential(
            OctaveCB(in_channels=inplanes,out_channels=planes * Bottleneck.expansion, kernel_size=1, stride=stride, padding=0)
        )
    layers = []
    layers.append(Bottleneck(inplanes, planes, stride, downsample))
    for _ in range(1, blocks-1):
        layers.append(Bottleneck(planes, planes))
    layers.append(BottleneckLast(planes, planes))


    return nn.Sequential(*layers)

class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class UpConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(UpConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(self.upsample(input))


@BACKBONES.register_module
class OctoveUnet(nn.Module):

    def __init__(self, in_ch, channels=16, blocks=3):
        super(OctoveUnet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = make_layer(channels, channels * 2, blocks, stride=2, First=True)
        self.layer2 = make_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer3 = make_layer(channels * 4, channels * 8, blocks, stride=2)
        
        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5_0 = DoubleConv(channels * 6, channels * 2)
        self.conv5_1 = DoubleConv(channels * 6, channels * 2)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6_0 = DoubleConv(channels * 3, channels * 1)
        self.conv6_1 = DoubleConv(channels * 3, channels * 1)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upconv7 = UpConv(channels * 1, channels * 1)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        
        up_5_0, up_5_1 = self.up5(c4[0]), self.up5(c4[1])
        merge5_0, merge5_1 = torch.cat([up_5_0, c3[0]], dim=1), torch.cat([up_5_1, c3[1]], dim=1)
        c5_0, c5_1 =  self.conv5_0(merge5_0), self.conv5_1(merge5_1)
        up_6_0, up_6_1 = self.up6(c5_0), self.up6(c5_1)
        merge6_0, merge6_1 = torch.cat([up_6_0, c2[0]], dim=1), torch.cat([up_6_1, c2[1]], dim=1)
        c6_0, c6_1 =  self.conv6_0(merge6_0), self.conv6_1(merge6_1)
        up_7_0, up_7_1 = self.up7(c6_0), self.upconv7(c6_1)
        merge7 = torch.cat([up_7_0, up_7_1, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        return up_8


