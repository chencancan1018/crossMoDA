import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss

softmax_helper = lambda x: F.softmax(x, 1)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1) # one-hot encode from gt-index

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list([ i for i in range(2, len(shp_x))])
        else:
            axes = list([ i for i in range(2, len(shp_x))])

        x = softmax_helper(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()
        # print("softdiceloss debug: ", dc, y.unique())

        return 1 - dc

@HEADS.register_module
class SegSoftHead(nn.Module):

    def __init__(self, in_channels, classes=5, is_aux=False, patch_size=(128,128,128)):
        super(SegSoftHead, self).__init__()

        self.conv = nn.Conv3d(in_channels, classes, 1)
        self.conv1 = nn.Conv3d(in_channels, classes, 1)
        self.conv2 = nn.Conv3d(2 * in_channels, classes, 1)
        self.conv3 = nn.Conv3d(4 * in_channels, classes, 1)
        self.multi_loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False, smooth=1e-5)
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
        
    def loss(self, inputs, target):
        seg_predict = inputs
        with torch.no_grad():
            seg = target 
            seg = seg[:, 0, ::].long()
            target[target <= 0] = 0
            target[target > 3] = 0
            
            radius = random.choice([7, 9, 11, 13, 15])
            lesion_target = (target > 0) * (target < 3) * 1.0
            hard_tn_seg = torch.max_pool3d(lesion_target, kernel_size= radius, stride=1, padding = radius//2)
            hard_tn_seg = hard_tn_seg * (1 - lesion_target)
            hard_tn_seg = hard_tn_seg[:, 0, ::].long()
            
            seg_weight = (15 * (hard_tn_seg > 0) + 1 * (seg == 0) + 5 * (seg == 1) + 15 * (seg == 2) + 5 * (seg == 3))
            # seg_weight = (1 * (seg == 0) + 5 * (seg == 1) + 15 * (seg == 2) + 5 * (seg == 3))
            seg_total_sum = ((seg < 0.5).sum() + (seg >= 0.5).sum() + 1)

            if self.is_aux:
                # patch_size = [128, 128, 128]  
                shape = [int(v // 2)  for v in self._patch_size]
                shape[0] = self._patch_size[0]
                target1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg1 = target1[:, 0, ::].long()
                seg1_weight = (1 * (seg1 == 0) + 5 * (seg1 == 1) + 15 * (seg1 == 2) + 5 * (seg1 == 3))
                seg1_total_sum = ((seg1 < 0.5).sum() + (seg1 >= 0.5).sum() + 1)

                shape = [int(v // 4)  for v in self._patch_size]
                shape[0] = self._patch_size[0]
                target2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg2 = target2[:, 0, ::].long()
                seg2_weight = (1 * (seg2 == 0) + 5 * (seg2 == 1) + 15 * (seg2 == 2) + 5 * (seg2 == 3))
                seg2_total_sum = ((seg2 < 0.5).sum() + (seg2 >= 0.5).sum() + 1)

                shape = [int(v // 8)  for v in self._patch_size]
                shape[0] = self._patch_size[0]
                target3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg3 = target3[:, 0, ::].long()
                seg3_weight = (1 * (seg3 == 0) + 5 * (seg3 == 1) + 15 * (seg3 == 2) + 5 * (seg3 == 3))
                seg3_total_sum = ((seg3 < 0.5).sum() + (seg3 >= 0.5).sum() + 1)

        if self.is_aux:
            loss = self.multi_loss_func(seg_predict[0], seg)
            dice_loss = self.dice_loss(seg_predict[0], target)
        else:
            loss = self.multi_loss_func(seg_predict, seg)
            dice_loss = self.dice_loss(seg_predict, target)

        dice_ = dice_loss
        ce_loss = (loss * seg_weight).sum() / seg_total_sum

        if self.is_aux:
            loss = self.multi_loss_func(seg_predict[1], seg1)
            dice_loss = self.dice_loss(seg_predict[1], target1)
            dice_ += (1/2) * dice_loss
            ce_loss += (1/2) * (loss * seg1_weight).sum() / seg1_total_sum

            loss = self.multi_loss_func(seg_predict[2], seg2)
            dice_loss = self.dice_loss(seg_predict[2], target2)
            dice_ += (1/4) * dice_loss
            ce_loss += (1/4) * (loss * seg2_weight).sum() / seg2_total_sum

            loss = self.multi_loss_func(seg_predict[3], seg3)
            dice_loss = self.dice_loss(seg_predict[3], target3)
            dice_ += (1/8) * dice_loss
            ce_loss += (1/8) * (loss * seg3_weight).sum() / seg3_total_sum

        return {'ce_loss': 20 * ce_loss, 'dice_loss': dice_}
    
    def mixloss(self, inputs, target, mix_target=None, lam=1.):
        seg_predict = inputs
        with torch.no_grad():
            seg = target[:, 0, ::].long()
            seg_weight = (1 * (seg == 0) + 5 * (seg == 1) + 5 * (seg == 2) + 5 * (seg == 3))
            seg_total_sum = ((seg < 0.5).sum() + (seg >= 0.5).sum() + 1)
            mix_seg = mix_target[:, 0, ::].long()
            mix_seg_weight = (1 * (mix_seg == 0) + 5 * (mix_seg == 1) + 5 * (mix_seg == 2) + 5 * (mix_seg == 3))
            mix_seg_total_sum = ((mix_seg < 0.5).sum() + (mix_seg >= 0.5).sum() + 1)

            target[target <= 0] = 0
            target[target > 3] = 0
            mix_target[mix_target <= 0] = 0
            mix_target[mix_target > 3] = 0 #0,1,2,3 to 0,1,2

            shape = [int(v // 2)  for v in self._patch_size]
            shape[0] = self._patch_size[0]
            target1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
            seg1 = target1[:, 0, ::].long()
            seg1_weight = (1 * (seg1 == 0) + 5 * (seg1 == 1) + 5 * (seg1 == 2) + 5 * (seg1 == 3))
            seg1_total_sum = ((seg1 < 0.5).sum() + (seg1 >= 0.5).sum() + 1)
            mix_target1 = torch.nn.functional.interpolate(mix_target.float(), size=shape, mode="nearest")
            mix_seg1 = mix_target1[:, 0, ::].long()
            mix_seg1_weight = (1 * (mix_seg1 == 0) + 5 * (mix_seg1 == 1) + 5 * (mix_seg1 == 2) + 5 * (mix_seg1 == 3))
            mix_seg1_total_sum = ((mix_seg1 < 0.5).sum() + (mix_seg1 >= 0.5).sum() + 1)

            shape = [int(v // 4)  for v in self._patch_size]
            shape[0] = self._patch_size[0]
            target2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
            seg2 = target2[:, 0, ::].long()
            seg2_weight = (1 * (seg2 == 0) + 5 * (seg2 == 1) + 5 * (seg2 == 2) + 5 * (seg2 == 3))
            seg2_total_sum = ((seg2 < 0.5).sum() + (seg2 >= 0.5).sum() + 1)
            mix_target2 = torch.nn.functional.interpolate(mix_target.float(), size=shape, mode="nearest")
            mix_seg2 = mix_target2[:, 0, ::].long()
            mix_seg2_weight = (1 * (mix_seg2 == 0) + 5 * (mix_seg2 == 1) + 5 * (mix_seg2 == 2) + 5 * (mix_seg2 == 3))
            mix_seg2_total_sum = ((mix_seg2 < 0.5).sum() + (mix_seg2 >= 0.5).sum() + 1)

            shape = [int(v // 8)  for v in self._patch_size]
            shape[0] = self._patch_size[0]
            target3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
            seg3 = target3[:, 0, ::].long()
            seg3_weight = (1 * (seg3 == 0) + 5 * (seg3 == 1) + 5 * (seg3 == 2) + 5 * (seg3 == 3))
            seg3_total_sum = ((seg3 < 0.5).sum() + (seg3 >= 0.5).sum() + 1)
            mix_target3 = torch.nn.functional.interpolate(mix_target.float(), size=shape, mode="nearest")
            mix_seg3 = mix_target3[:, 0, ::].long()
            mix_seg3_weight = (1 * (mix_seg3 == 0) + 5 * (mix_seg3 == 1) + 5 * (mix_seg3 == 2) + 5 * (mix_seg3 == 3))
            mix_seg3_total_sum = ((mix_seg3 < 0.5).sum() + (mix_seg3 >= 0.5).sum() + 1)

        loss = self.multi_loss_func(seg_predict[0], seg)
        ce_loss = (loss * seg_weight).sum() / seg_total_sum
        mix_loss = self.multi_loss_func(seg_predict[0], mix_seg)
        mix_ce_loss = (mix_loss * mix_seg_weight).sum() / mix_seg_total_sum
        ce_loss = ce_loss * lam + mix_ce_loss * (1 - lam)

        dice_loss = self.dice_loss(seg_predict[0], target)
        mix_dice_loss = self.dice_loss(seg_predict[0], mix_target)
        dice_ = dice_loss * lam + mix_dice_loss * (1 - lam)

        loss = self.multi_loss_func(seg_predict[1], seg1)
        dice_loss = self.dice_loss(seg_predict[1], target1)
        ce_loss += (1/2) * lam * (loss * seg1_weight).sum() / seg1_total_sum
        dice_ += (1/2) * lam * dice_loss
        loss = self.multi_loss_func(seg_predict[1], mix_seg1)
        dice_loss = self.dice_loss(seg_predict[1], mix_target1)
        ce_loss += (1/2) * (1-lam) * (loss * mix_seg1_weight).sum() / mix_seg1_total_sum
        dice_ += (1/2) * (1-lam) * dice_loss

        loss = self.multi_loss_func(seg_predict[2], seg2)
        dice_loss = self.dice_loss(seg_predict[2], target2)
        dice_ += (1/4) * lam * dice_loss
        ce_loss += (1/4) * lam * (loss * seg2_weight).sum() / seg2_total_sum
        loss = self.multi_loss_func(seg_predict[2], mix_seg2)
        dice_loss = self.dice_loss(seg_predict[2], mix_target2)
        ce_loss += (1/4) * (1-lam) * (loss * mix_seg2_weight).sum() / mix_seg2_total_sum
        dice_ += (1/4) * (1-lam) * dice_loss

        loss = self.multi_loss_func(seg_predict[3], seg3)
        dice_loss = self.dice_loss(seg_predict[3], target3)
        dice_ += (1/8) * lam * dice_loss
        ce_loss += (1/8) * lam * (loss * seg3_weight).sum() / seg3_total_sum
        loss = self.multi_loss_func(seg_predict[3], mix_seg3)
        dice_loss = self.dice_loss(seg_predict[3], mix_target3)
        ce_loss += (1/8) * (1-lam) * (loss * mix_seg3_weight).sum() / mix_seg3_total_sum
        dice_ += (1/8) * (1-lam) * dice_loss

        return {'ce_loss': 5 * ce_loss, 'dice_loss': dice_}



