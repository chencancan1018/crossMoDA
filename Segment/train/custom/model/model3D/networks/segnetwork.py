
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from starship.umtf.common import build_pipelines
from starship.umtf.common.model import NETWORKS, build_backbone, build_head

p_mixup = 0.5
def mixup(input, mask, clip=[0,1]):
    indices = torch.randperm(input.size(0))
    shuffle_input = input[indices]
    shuffle_mask = mask[indices]
    lam = np.random.beta(0.4, 0.4)
    lam = np.clip(lam, clip[0], clip[1])
    mix_input = input * lam + shuffle_input * (1 - lam)
    return mix_input, shuffle_mask, lam

@NETWORKS.register_module
class SegNetwork(nn.Module):

    def __init__(self, backbone,
                head, apply_sync_batchnorm=False,
                train_cfg=None,
                test_cfg=None
                ):
        super(SegNetwork, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self._show_count = 0
        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, vol, seg):
        vol = vol.float()
        seg = seg.float()
        features = self.backbone(vol)
        head_outs = self.head(features)
        loss = self.head.loss(head_outs, seg)
        return loss

    # @torch.jit.ignore
    # def forward(self, vol, seg):
    #     vol = vol.float()
    #     seg = seg.float()

    #     do_mixup = False
    #     if random.random() < p_mixup:
    #         do_mixup = True
    #         mix_vol, mix_seg, lam = mixup(vol, seg)
    #     if do_mixup:
    #         features = self.backbone(mix_vol)
    #         head_outs = self.head(features)
    #         loss = self.head.mixloss(head_outs, seg, mix_target=mix_seg, lam=lam)
    #     else:
    #         features = self.backbone(vol)
    #         head_outs = self.head(features)
    #         loss = self.head.loss(head_outs, seg)
    #     return loss

    @torch.jit.export
    def forward_test(self, img):
        features = self.backbone(img)
        seg_predict = self.head(features)
        return seg_predict


    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)
