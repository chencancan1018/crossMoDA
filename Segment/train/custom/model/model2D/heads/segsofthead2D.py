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

        return 1 - dc

@HEADS.register_module
class SegSoftHead2D(nn.Module):

    def __init__(self, in_channels, classes=5, is_aux=False, patch_size=(256, 256)):
        super(SegSoftHead2D, self).__init__()

        self.conv = nn.Conv2d(in_channels, classes, 1)
        self.conv1 = nn.Conv2d(in_channels, classes, 1)
        self.conv2 = nn.Conv2d(2 * in_channels, classes, 1)
        self.conv3 = nn.Conv2d(4 * in_channels, classes, 1)
        self.multi_loss_func = torch.nn.CrossEntropyLoss(reduce=False)
        self.dice_loss = SoftDiceLoss(batch_dice=True, do_bg=False, smooth=1e-5)
        self._patch_size = patch_size
        self.is_aux = is_aux

    def forward(self, inputs):
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

            seg_tp = (seg >= 0.5) + (seg >= 1.5) * 9
            seg_tn = (seg < 0.5) * 1 #+ margin_region * 3
            seg_tp_sum = ((seg >= 0.5).sum() + 1)
            seg_tn_sum = ((seg < 0.5).sum() + 1)

            if self.is_aux:
                # patch_size = [256, 256]  
                shape = tuple([int(v // 2)  for v in self._patch_size])
                seg1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")[:, 0, ::].long()
                target1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg_tp1 = (seg1 >= 0.5)
                seg_tn1 = (seg1 < 0.5) * 1 
                seg_tp_sum1 = ((seg1 >= 0.5).sum() + 1)
                seg_tn_sum1 = ((seg1 < 0.5).sum() + 1)

                shape = tuple([int(v // 4)  for v in self._patch_size])
                seg2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")[:, 0, ::].long()
                target2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg_tp2 = (seg2 >= 0.5) 
                seg_tn2 = (seg2 < 0.5) * 1 
                seg_tp_sum2 = ((seg2 >= 0.5).sum() + 1)
                seg_tn_sum2 = ((seg2 < 0.5).sum() + 1)

                shape = tuple([int(v // 8)  for v in self._patch_size])
                seg3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")[:, 0, ::].long()
                target3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest")
                seg_tp3 = (seg3 >= 0.5) 
                seg_tn3 = (seg3 < 0.5) * 1 
                seg_tp_sum3 = ((seg3 >= 0.5).sum() + 1)
                seg_tn_sum3 = ((seg3 < 0.5).sum() + 1)


        if self.is_aux:
            loss = self.multi_loss_func(seg_predict[0], seg)
            dice_loss = self.dice_loss(seg_predict[0], target)
        else:
            loss = self.multi_loss_func(seg_predict, seg)
            dice_loss = self.dice_loss(seg_predict, target)
        loss_pos = (loss * seg_tp).sum() / seg_tp_sum
        loss_neg = (loss * seg_tn).sum() / seg_tn_sum
        pos_ = loss_pos
        neg_ = loss_neg
        dice_ = dice_loss

        if self.is_aux:
            loss = self.multi_loss_func(seg_predict[1], seg1)
            loss_pos = (loss * seg_tp1).sum() / seg_tp_sum1
            loss_neg = (loss * seg_tn1).sum() / seg_tn_sum1
            dice_loss = self.dice_loss(seg_predict[1], target1)
            pos_ += (1/2) * loss_pos
            neg_ += (1/2) * loss_neg
            dice_ += (1/2) * dice_loss

            loss = self.multi_loss_func(seg_predict[2], seg2)
            loss_pos = (loss * seg_tp2).sum() / seg_tp_sum2
            loss_neg = (loss * seg_tn2).sum() / seg_tn_sum2
            dice_loss = self.dice_loss(seg_predict[2], target2)
            pos_ += (1/4) * loss_pos
            neg_ += (1/4) * loss_neg
            dice_ += (1/4) * dice_loss

            loss = self.multi_loss_func(seg_predict[3], seg3)
            loss_pos = (loss * seg_tp3).sum() / seg_tp_sum3
            loss_neg = (loss * seg_tn3).sum() / seg_tn_sum3
            dice_loss = self.dice_loss(seg_predict[3], target3)
            pos_ += (1/8) * loss_pos
            neg_ += (1/8) * loss_neg
            dice_ += (1/8) * dice_loss

        return {'loss_pos': pos_, 'loss_neg': neg_, 'dice_loss': dice_}


