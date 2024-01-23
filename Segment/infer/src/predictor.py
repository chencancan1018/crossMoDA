
import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from scipy.ndimage import zoom
import torch
import torch.nn as nn

class SegConfig:

    def __init__(self,  network_f):
        # TODO: 模型配置文件
        self.network_f = network_f
        if self.network_f is not None:
            from mmcv import Config

            if isinstance(self.network_f, str):
                self.network_cfg = Config.fromfile(self.network_f)
            else:
                import tempfile

                with tempfile.TemporaryDirectory() as temp_config_dir:
                    with tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix='.py') as temp_config_file:
                        with open(temp_config_file.name, 'wb') as f:
                            f.write(self.network_f.read())

                        self.network_cfg = Config.fromfile(temp_config_file.name)

    def __repr__(self) -> str:
        return str(self.__dict__)


class SegModel:

    def __init__(self, model_f, network_f):
        # TODO: 模型文件定制
        self.model_f = model_f
        self.network_f = network_f


class SegPredictor:

    def __init__(self, gpu: int, model: SegModel):
        self.gpu = gpu
        self.model = model
        self.config = SegConfig(self.model.network_f)
        self.load_model()

    def load_model(self):
        self.net = self._load_model(self.model.model_f, self.config.network_cfg, half=False)

    def _load_model(self, model_f, network_f, half=False) -> None:
        if isinstance(model_f, str):
            # 根据后缀判断类型
            net = self.load_model_pth(model_f, network_f, half)
        return net
    
    def load_model_pth(self, model_f, network_cfg, half) -> None:
        # 加载动态图
        config = network_cfg

        backbone = ResUnet(config.in_ch, channels=config.model["backbone"]["channels"])
        head = SegSoftHead(config.model["head"]["in_channels"], classes=config.model["head"]["classes"])
        net = SegNetwork(backbone, head)

        checkpoint = torch.load(model_f, map_location=f"cpu")
        net.load_state_dict(checkpoint["state_dict"], strict=False)
        net.eval()
        # if half:
        net.half()
        net.cuda(self.gpu)
        net = net.forward_test
        return net

    def _get_input(self, vol):

        config = self.config.network_cfg
        vol_shape = np.array(vol.shape)
        patch_size = np.array(config.patch_size)
        # if np.any(vol_shape != patch_size):
        #     vol = zoom(vol, np.array(patch_size/vol_shape), order=1)
        assert (vol_shape == patch_size).all()
        vol = torch.from_numpy(vol).float()[None, None]
        return vol

    def forward(self, vol):

        config = self.config.network_cfg
        patch_size = np.array(config.patch_size)
        ori_shape = np.array(vol.shape)

        with autocast():
            data = self._get_input(vol)
            data = data.cuda(self.gpu).detach()
            pred_seg = self.net(data)
            del data
            if pred_seg.size()[1] > 1:
                pred_seg = F.softmax(pred_seg, dim=1)
                pred_seg = torch.argmax(pred_seg, dim=1, keepdim=True)
            else:
                pred_seg = torch.sigmoid(pred_seg)
                pred_seg[pred_seg >= config.threshold] = 1
                pred_seg[pred_seg < config.threshold] = 0
        
        heatmap = pred_seg.cpu().detach().numpy()[0, 0].astype(np.int8)
        heatmap = heatmap.astype(np.uint8)
        return heatmap


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

class SegNetwork(nn.Module):

    def __init__(self, backbone,
                head, apply_sync_batchnorm=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(SegNetwork, self).__init__()
        self.backbone = backbone
        self.head = head

    @torch.jit.ignore
    def forward(self, vol, seg):
        vol = vol.float()
        seg = seg.float()
        features = self.backbone(vol)
        head_outs = self.head(features)
        loss = self.head.loss(head_outs, seg)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        features = self.backbone(img)
        seg_predict = self.head(features)
        return seg_predict

class SegSoftHead(nn.Module):

    def __init__(self, in_channels, classes=5, is_aux=False, patch_size=(128,128,128)):
        super(SegSoftHead, self).__init__()

        self.conv = nn.Conv3d(in_channels, classes, 1)
        self.conv1 = nn.Conv3d(in_channels, classes, 1)
        self.conv2 = nn.Conv3d(2 * in_channels, classes, 1)
        self.conv3 = nn.Conv3d(4 * in_channels, classes, 1)
        self._patch_size = patch_size
        self.is_aux = is_aux

    def forward(self, inputs):
        # inputs = F.interpolate(inputs, scale_factor=1.0, mode='trilinear')
        convlayers = [self.conv, self.conv1, self.conv2, self.conv3]
        if self.is_aux:
            predicts = list()
            for idx in range(len(inputs)):
                predicts.append(convlayers[idx](inputs[idx]))
        else:
            predicts = self.conv(inputs)
        return predicts

    def forward_test(self, inputs):
        predicts = self.conv(inputs)
        return predicts



