import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import NETWORKS, build_backbone, BACKBONES


def _ASPPConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    asppconv = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )
    return asppconv

class ASPP(nn.Module):
    """
    ASPP module in `DeepLabV3, see also in <https://arxiv.org/abs/1706.05587>` 
    """
    def __init__(self, in_channels, out_channels, output_stride=16):
        super(ASPP, self).__init__()

        if output_stride == 16:
            astrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            astrous_rates = [0, 12, 24, 36]
        else:
            raise Warning('Output stride must be 8 or 16!')

        # astrous spational pyramid pooling part
        self.conv1 = _ASPPConv(in_channels, out_channels, 1, 1)
        self.conv2 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[1], dilation=astrous_rates[1])
        self.conv3 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[2], dilation=astrous_rates[2])
        self.conv4 = _ASPPConv(in_channels, out_channels, 3, 1, padding=astrous_rates[3], dilation=astrous_rates[3])

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, input):
        input1 = self.conv1(input)
        input2 = self.conv2(input)
        input3 = self.conv3(input)
        input4 = self.conv4(input)
        
        input5 = F.interpolate(self.pool(input), size=input4.size()[2:], mode='trilinear', align_corners=False)
        output = torch.cat((input1, input2, input3, input4, input5), dim=1)
        output = self.bottleneck(output)
        return output

class Decoder(nn.Module):
    def __init__(self, high_in, high_out, low_in, low_out):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv3d(low_in, low_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(low_out)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(low_out + high_in, high_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(high_out)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv3d(high_out, high_out, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(high_out)
        self.dropout4 = nn.Dropout(0.5)

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.relu(self.bn1(low_level_feature))
        
        x_4 = F.interpolate(x, size=low_level_feature.size()[2:], mode='trilinear', align_corners=False)
        x_4_cat = torch.cat([x_4, low_level_feature], dim=1)
        x_4_cat = self.relu(self.bn2(self.conv2(x_4_cat)))
        x_4_cat = self.relu(self.bn3(self.conv3(x_4_cat)))
        return x_4_cat

@BACKBONES.register_module
class DeepLab(nn.Module):
    """
    The implementation of DeepLabV3, see also in <https://arxiv.org/abs/1706.05587>
    config:
        backbone=dict(type=`'DeepLab'`, 
                    backbone=`'ResNet3D_101'`,
                    in_ch=in_ch, 
                    in_chanels=512, 
                    out_channels=256,
                    low_in=64, 
                    low_out=64)
    """
    def __init__(self, backbone, in_channels, out_channels, low_in, low_out, output_stride=16, contrast=True):
        super(DeepLab, self).__init__()
        self.backbone = build_backbone(backbone)
        self.aspp = ASPP(in_channels, in_channels, output_stride=output_stride)
        self.decoder = Decoder(in_channels, out_channels, low_in, low_out)
        self.contrast = contrast
        # if not contrast:
        #     for module in [self.backbone, self.aspp]:
        #         for p in module.parameters():
        #             p.requires_grad = False
    
    def forward(self, input):
        x, low_level_feature = self.backbone(input)
        x = self.aspp(x)

        output = self.decoder(x, low_level_feature)
        output = F.interpolate(output, size=input.size()[2:], mode='trilinear', align_corners=False)
        return output