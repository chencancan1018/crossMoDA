import torch
import torch.nn as nn
from starship.umtf.common.model import BACKBONES


char = 'instance'
if char == 'instance':
    norm_type = nn.InstanceNorm3d
    use_bias = True
else:
    norm_type = nn.BatchNorm3d
    use_bias = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=use_bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=use_bias)


    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1,downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride) 
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_type(planes)
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


def make_res_layer(inplanes, planes, blocks,  kernel_size=3, stride=1, padding=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        norm_type(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, kernel_size, stride, padding, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2), bias=use_bias),
            norm_type(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1, bias=use_bias),
            norm_type(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)

class DoubleConvDown(nn.Module):

    def __init__(self, in_ch, out_ch, stride=(1,2,2), kernel_size=(1,3,3), padding=(0,1,1)):
        super(DoubleConvDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias),
            norm_type(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1, bias=use_bias),
            norm_type(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)



@BACKBONES.register_module
class ResUnet(nn.Module):

    def __init__(self, in_ch, channels=16, blocks=3, is_aux=False):
        super(ResUnet, self).__init__()

        self.in_conv = DoubleConvDown(in_ch, channels, stride=(1,2,2), kernel_size=(1,3,3), padding=(0,1,1))
        self.layer1 = make_res_layer(channels, channels * 2, blocks, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.is_aux = is_aux

        self.up5 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)

    def forward(self, input):
        c1 = self.in_conv(input)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        if self.is_aux:
            return [up_8, c7, c6, c5]
        else:
            return up_8



if __name__ == '__main__':
    model = ResUnet(3, 32)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    # classical ResUnet (downsample=16, channel=16) params: 3796816
    # classical ResUnet (downsample=16, channels=32) params: 15176864
    # MW ResUnet (downsample=32, channels=16) params: 15176864
