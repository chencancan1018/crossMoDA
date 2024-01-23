from turtle import forward
import torch
import torch.nn as nn
from starship.umtf.common.model import BACKBONES

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm=None):
        super(BasicBlock, self).__init__()
        if norm is None:
            self.bn1 = nn.BatchNorm3d(planes)
            self.bn2 = nn.BatchNorm3d(planes)
        elif norm == 'instance':
            self.bn1 = nn.InstanceNorm3d(planes)
            self.bn2 = nn.InstanceNorm3d(planes)
        else:
            raise KeyError(" the norm is not batch norm and instance norm!!")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DecodeBlock(nn.Module):
    def __init__(self, in_planes, out_planes, upsample_kernel_size, norm_name):
        super(DecodeBlock, self).__init__()
        self.transp_conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=upsample_kernel_size, stride=upsample_kernel_size)
        downsample = nn.Sequential(nn.Conv3d(out_planes * 2, out_planes, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm3d(out_planes),
                                )
        self.block = BasicBlock(out_planes * 2, out_planes, stride=1, norm=norm_name, downsample=downsample)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.block(out)
        return out


@BACKBONES.register_module
class UNETR3D(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name='instance',
        dropout_rate = 0.0,
    ):
        super().__init__()
        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        from .vit import ViT
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=3
        )

        self.encoder1 = BasicBlock(in_channels, feature_size, stride=1, norm=norm_name)
        self.encoder2 = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, feature_size * 2, kernel_size=2, stride=2),
            nn.ConvTranspose3d(feature_size * 2, feature_size * 2, kernel_size=2, stride=2),
            BasicBlock(feature_size * 2, feature_size * 2, stride=1, norm=norm_name),
            nn.ConvTranspose3d(feature_size * 2, feature_size * 2, kernel_size=2, stride=2),
            BasicBlock(feature_size * 2, feature_size * 2, stride=1, norm=norm_name),
        )
        self.encoder3 = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, feature_size * 4, kernel_size=2, stride=2),
            nn.ConvTranspose3d(feature_size * 4, feature_size * 4, kernel_size=2, stride=2),
            BasicBlock(feature_size * 4, feature_size * 4, stride=1, norm=norm_name),
        )
        self.encoder4 = nn.Sequential(
            nn.ConvTranspose3d(hidden_size, feature_size * 8, kernel_size=2, stride=2),
        )

        self.decoder5 = DecodeBlock(hidden_size, feature_size * 8, 2, norm_name)
        self.decoder4 = DecodeBlock(feature_size * 8, feature_size * 4, 2, norm_name)
        self.decoder3 = DecodeBlock(feature_size * 4, feature_size * 2, 2, norm_name)
        self.decoder2 = DecodeBlock(feature_size * 2, feature_size * 1, 2, norm_name)

        # self.out = nn.Conv3d(feature_size, out_channels, kernel_size=1, stride=1, dropout=dropout_rate, bias=True)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        # print('x: ', x.size())
        enc1 = self.encoder1(x_in)
        # print('enc1: ', enc1.size())
        x2 = hidden_states_out[3]
        # print('x2: ', x2.size())
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        # print('enc2: ', enc2.size())
        x3 = hidden_states_out[6]
        # print('x3: ', x3.size())
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        # print('enc3: ', enc3.size())
        x4 = hidden_states_out[9]
        # print('x4: ', x4.size())
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        # print('enc4: ', enc4.size())
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        # print('dec4: ', dec4.size())
        dec3 = self.decoder5(dec4, enc4)
        # print('dec3: ', dec3.size())
        dec2 = self.decoder4(dec3, enc3)
        # print('dec2: ', dec2.size())
        dec1 = self.decoder3(dec2, enc2)
        # print('dec1: ', dec1.size())
        out = self.decoder2(dec1, enc1)
        # print('out: ', out.size())
        return out
