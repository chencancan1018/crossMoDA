import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from starship.umtf.common.model import HEADS, LOSSES, build_loss

@HEADS.register_module
class SegSigHead(nn.Module):

    def __init__(self, in_channels, is_aux=False, patch_size=(128,128,128)):
        super(SegSigHead, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.conv1 = nn.Conv3d(in_channels, 1, 1)
        self.conv2 = nn.Conv3d(2 * in_channels, 1, 1)
        self.conv3 = nn.Conv3d(4 * in_channels, 1, 1)
        self.bce_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)
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
    
    def _dice_loss(self, logits, bin_target):
        eps = 1e-9
        pred = torch.sigmoid(logits)
        inter_section = (pred * bin_target).sum()
        inter_section = 2 * inter_section + eps
        union = pred.sum() + bin_target.sum() + eps
        dice = inter_section / union
        return 1 - dice
        
    def loss(self, inputs, target):
        seg_predict = inputs
        with torch.no_grad():
            seg = target 
            seg = seg.float()

            # hard negtive mining
            # radius = random.sample([5, 7, 9, 13], 1)[0]
            # target_binary = (target > 0.5) * 1.0
            # target_dilation = torch.max_pool3d(target_binary, kernel_size=radius, stride=1, padding=radius // 2)
            # margin_region = target_dilation * ( 1 - target_binary)
            seg_tp = (seg >= 0.5) 
            seg_tn = (seg < 0.5) * 1 #+ margin_region * 3
            seg_tp_sum = ((seg >= 0.5).sum() + 1)
            seg_tn_sum = ((seg < 0.5).sum() + 1)

            if self.is_aux:
                # patch_size = [128, 128, 128]  
                shape = tuple([int(v // 2)  for v in self._patch_size])
                seg1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
                target1 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
                seg_tp1 = (seg1 >= 0.5) 
                seg_tn1 = (seg1 < 0.5) * 1 
                seg_tp_sum1 = ((seg1 >= 0.5).sum() + 1)
                seg_tn_sum1 = ((seg1 < 0.5).sum() + 1)

                shape = tuple([int(v // 4)  for v in self._patch_size])
                seg2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
                target2 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
                seg_tp2 = (seg2 >= 0.5) 
                seg_tn2 = (seg2 < 0.5) * 1 
                seg_tp_sum2 = ((seg2 >= 0.5).sum() + 1)
                seg_tn_sum2 = ((seg2 < 0.5).sum() + 1)

                shape = tuple([int(v // 8)  for v in self._patch_size])
                seg3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
                target3 = torch.nn.functional.interpolate(target.float(), size=shape, mode="nearest").float()
                seg_tp3 = (seg3 >= 0.5) 
                seg_tn3 = (seg3 < 0.5) * 1 
                seg_tp_sum3 = ((seg3 >= 0.5).sum() + 1)
                seg_tn_sum3 = ((seg3 < 0.5).sum() + 1)

        if self.is_aux:
            loss = self.bce_loss_func(seg_predict[0], seg)
            dice_loss = self._dice_loss(seg_predict[0], target)
        else:
            loss = self.bce_loss_func(seg_predict, seg)
            dice_loss = self._dice_loss(seg_predict, target)

        loss_pos = (loss * seg_tp).sum() / seg_tp_sum
        loss_neg = (loss * seg_tn).sum() / seg_tn_sum
        
        pos_ = loss_pos
        neg_ = loss_neg
        dice_ = dice_loss

        if self.is_aux:
            loss = self.bce_loss_func(seg_predict[1], seg1)
            loss_pos = (loss * seg_tp1).sum() / seg_tp_sum1
            loss_neg = (loss * seg_tn1).sum() / seg_tn_sum1
            dice_loss = self._dice_loss(seg_predict[1], target1)
            pos_ += (1/2) * loss_pos
            neg_ += (1/2) * loss_neg
            dice_ += (1/2) * dice_loss

            loss = self.bce_loss_func(seg_predict[2], seg2)
            loss_pos = (loss * seg_tp2).sum() / seg_tp_sum2
            loss_neg = (loss * seg_tn2).sum() / seg_tn_sum2
            dice_loss = self._dice_loss(seg_predict[2], target2)
            pos_ += (1/4) * loss_pos
            neg_ += (1/4) * loss_neg
            dice_ += (1/4) * dice_loss

            loss = self.bce_loss_func(seg_predict[3], seg3)
            loss_pos = (loss * seg_tp3).sum() / seg_tp_sum3
            loss_neg = (loss * seg_tn3).sum() / seg_tn_sum3
            dice_loss = self._dice_loss(seg_predict[3], target3)
            pos_ += (1/8) * loss_pos
            neg_ += (1/8) * loss_neg
            dice_ += (1/8) * dice_loss

        return {'loss_pos': pos_, 'loss_neg': neg_, 'dice_loss': dice_}
