import torch.nn as nn
import torch
import torch.nn.functional as F

from starship.umtf.common.model import BACKBONES

class SKConv2D(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv2D, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x.size(), (8, 32, 64,64,64)
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1) # feas, (8,2,32,64,64,64)
        fea_U = torch.sum(feas, dim=1) # fea_U, (8, 32,64,64,64)
        # fea_s = self.gap(fea_U).squeeze_() 
        fea_s = fea_U.mean(-1).mean(-1) # fea_s, (8,32)
        fea_z = self.fc(fea_s)  # fea_z, (8,32)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1) # attention_vectors, (8,2, 32)
        attention_vectors = self.softmax(attention_vectors)  # attention_vectors, (8,2,32)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1) # attention_vectors, (8,2,32,1,1,1)
        fea_v = (feas * attention_vectors).sum(dim=1) # fea_v, (8, 32,64,64,64)
        return fea_v

class SKConv3D(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=64):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv3D, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv3d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm3d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v



class SKUnit2D(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit2D, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv2D(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)

class SKUnit3D(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs, default is 2.
            G: num of convolution groups.
            r: the radio for compute d, the length of z, default is 2.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit3D, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv3d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm3d(mid_features),
            SKConv3D(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm3d(mid_features),
            nn.Conv3d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm3d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm3d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)



class conv_block(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1, no_bn=False):
        super(conv_block, self).__init__()
        if no_bn:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride,
                          padding=p_size,
                          dilation=dilation),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride,
                          padding=p_size,
                          dilation=dilation),
                nn.BatchNorm3d(chann_out),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class SKBranch(nn.Module):
    def __init__(self):
        super(SKBranch, self).__init__()

        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        self.layer7 = conv_block(64, 512, 1, stride=1, p_size=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out


# @BACKBONES.register_module
class SKNet(nn.Module):
    def __init__(self, d_count, c_count=1):
        super(SKNet, self).__init__()
        self.branch0 = SKBranch()
        self.branch1 = SKBranch()
        self.branch2 = SKBranch()
        self.layer8 = conv_block(3 * 512, 2048, 1, stride=1, p_size=0)
        self.layer8_extra = conv_block(2048, 2048, 1, stride=1, p_size=0)
        self.layerD = nn.Conv3d(in_channels=2048, out_channels=d_count, kernel_size=1, stride=1, padding=0)
        self.layer9 = conv_block(d_count, 256, 1, stride=1, p_size=0)
        self.layer10 = conv_block(256, 128, 1, stride=1, p_size=0)
        self.layerC = nn.Conv3d(in_channels=128, out_channels=c_count, kernel_size=1, stride=1, padding=0)

    def forward(self, img0, img1, img2):
        feature0 = self.branch0(img0)
        feature1 = self.branch1(img1)
        feature2 = self.branch2(img2)
        feature = torch.cat([feature0, feature1, feature2], dim=1)
        feature = self.layer8(feature)
        feature = F.dropout(feature, p=0.5)
        feature = self.layer8_extra(feature)
        d_predict = self.layerD(feature)
        feature = self.layer9(d_predict)
        feature = self.layer10(feature)
        c_predict = self.layerC(feature)
        return d_predict.reshape((d_predict.size(0), -1)), c_predict.reshape((c_predict.size(0), -1)),

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

@BACKBONES.register_module
class SKUnet(nn.Module):

    def __init__(self, in_ch, channels=16):
        super(SKUnet, self).__init__()

        self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3)
        self.layer1 = nn.Sequential(
            SKUnit3D(channels, channels * 2, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit3D(channels * 2, channels * 2, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit3D(channels * 2, channels * 2, 32, 2, 8, 2),
            nn.ReLU()
        ) 
        self.layer2 = nn.Sequential(
            SKUnit3D(channels * 2, channels * 4, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit3D(channels * 4, channels * 4, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit3D(channels * 4, channels * 4, 32, 2, 8, 2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            SKUnit3D(channels * 4, channels * 8, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            SKUnit3D(channels * 8, channels * 8, 32, 2, 8, 2),
            nn.ReLU(),
            SKUnit3D(channels * 8, channels * 8, 32, 2, 8, 2),
            nn.ReLU()
        )
        # self.layer1 = make_res_layer(channels, channels * 2, blocks, stride=2)
        # self.layer2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        # self.layer3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv5 = DoubleConv(channels * 12, channels * 4)
        self.up6 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv6 = DoubleConv(channels * 6, channels * 2)
        self.up7 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = DoubleConv(channels * 3, channels)
        self.up8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

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
        return up_8

if __name__ == '__main__':
    model = SKUnet(3, 32)
    print('#generator parameters:', sum(param.numel() for param in model.parameters()))
    # SKUnet (downsample=16, channel=16) params: 1005504
    # SKUnet (downsample=16, channel=32) params: 3975360
