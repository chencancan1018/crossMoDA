# import torch
# import torch.nn as nn
# from starship.umtf.common.model import BACKBONES
# import spconv.pytorch as spconv
# import torch.nn.functional as F


# # def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
# #     """3x3 convolution with padding."""
# #     return nn.Conv3d(
# #         in_planes,
# #         out_planes,
# #         kernel=3,
# #         stride=stride,
# #         padding=dilation,
# #         groups=groups,
# #         bias=False,
# #         dilation=dilation,
# #     )
# #
# #
# # def conv1x1(in_planes, out_planes, stride=1):
# #     """1x1 convolution."""
# #     return nn.Conv3d(in_planes, out_planes, kernel=1, stride=stride, bias=False)


# class BasicBlock(spconv.SparseModule):
#     expansion = 1

#     def __init__(self, inplanes, planes, indice_key=None):
#         super(BasicBlock, self).__init__()
#         self.conv = spconv.SparseSequential(
#             spconv.SubMConv3d(inplanes, planes, 3, padding=1, bias=False, indice_key=indice_key),
#             nn.BatchNorm1d(planes),
#             nn.ReLU(inplace=True),
#             spconv.SubMConv3d(inplanes, planes, 3, padding=1, bias=False, indice_key=indice_key),
#             nn.BatchNorm1d(planes))

#     def forward(self, x):
#         identity = x
#         out = self.conv(x)
#         out = spconv.AddTable()([out, identity])
#         # out = self.relu(out)
#         out = out.replace_feature(F.relu(out.features, inplace=True))

#         return out


# def make_res_layer(inplanes, planes, blocks, stride=1, indice_key=(None, None)):
#     downsample = spconv.SparseSequential(
#         spconv.SparseConv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
#                             indice_key=indice_key[0]),
#         nn.BatchNorm1d(planes),
#         nn.ReLU(inplace=True)
#     )

#     layers = []
#     layers.append(downsample)
#     for _ in range(1, blocks):
#         layers.append(BasicBlock(planes, planes, indice_key=indice_key[1]))

#     return spconv.SparseSequential(*layers)


# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, indice_key=(None, None)):
#         super(DoubleConv, self).__init__()
#         if stride >= 2:
#             self.conv = spconv.SparseSequential(
#                 spconv.SparseConv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2),
#                                     bias=False,
#                                     indice_key=indice_key[0]),
#                 nn.BatchNorm1d(out_ch),
#                 nn.ReLU(inplace=True),
#                 spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key[1]),
#                 nn.BatchNorm1d(out_ch),
#                 nn.ReLU(inplace=True),
#             )
#         else:
#             self.conv = spconv.SparseSequential(
#                 spconv.SubMConv3d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2),
#                                   bias=False,
#                                   indice_key=indice_key[0]),
#                 nn.BatchNorm1d(out_ch),
#                 nn.ReLU(inplace=True),
#                 spconv.SubMConv3d(out_ch, out_ch, 3, padding=1, bias=False, indice_key=indice_key[1]),
#                 nn.BatchNorm1d(out_ch),
#                 nn.ReLU(inplace=True),
#             )

#     def forward(self, input):
#         return self.conv(input)


# # class SingleConv(nn.Module):
# #     def __init__(self, in_ch, out_ch):
# #         super(SingleConv, self).__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)
# #         )
# #
# #     def forward(self, input):
# #         return self.conv(input)


# @BACKBONES.register_module
# class SpResUnet(nn.Module):
#     def __init__(self, in_ch, channels=16, blocks=3, gat_count=10):
#         super(SpResUnet, self).__init__()

#         self.in_conv = DoubleConv(in_ch, channels, stride=2, kernel_size=3, indice_key=('spconv0', 'subm0'))
#         self.f1 = make_res_layer(channels, channels * 2, blocks, stride=2, indice_key=('spconv1', 'subm1'))
#         self.f2 = make_res_layer(channels * 2, channels * 4, blocks, stride=2, indice_key=('spconv2', 'subm2'))
#         self.f3 = make_res_layer(channels * 4, channels * 8, blocks, stride=2, indice_key=('spconv3', 'subm3'))
#         self.f4 = make_res_layer(channels * 8, channels * 8, blocks, stride=2, indice_key=('spconv4', 'subm4'))
#         self.f5 = make_res_layer(channels * 8, channels * 8, blocks, stride=2, indice_key=('spconv5', 'subm5'))
#         self._gat_count = gat_count
#         self.up1 = spconv.SparseSequential(
#             spconv.SparseInverseConv3d(channels * 8, channels * 8, 3, indice_key='spconv5', bias=False),
#             nn.BatchNorm1d(channels * 8),
#             nn.ReLU(True)
#         )
#         self.b1 = DoubleConv(channels * 16, channels * 8, indice_key=('subm4', 'subm4'))
#         self.up2 = spconv.SparseSequential(
#             spconv.SparseInverseConv3d(channels * 8, channels * 8, 3, indice_key='spconv4', bias=False),
#             nn.BatchNorm1d(channels * 8),
#             nn.ReLU(True)
#         )
#         self.b2 = DoubleConv(channels * 16, channels * 8, indice_key=('subm3', 'subm3'))
#         self.up3 = spconv.SparseSequential(
#             spconv.SparseInverseConv3d(channels * 8, channels * 8, 3, indice_key='spconv3', bias=False),
#             nn.BatchNorm1d(channels * 8),
#             nn.ReLU(True)
#         )
#         self.b3 = DoubleConv(channels * 12, channels * 4, indice_key=('subm2', 'subm2'))

#         self.up4 = spconv.SparseSequential(
#             spconv.SparseInverseConv3d(channels * 4, channels * 4, 3, indice_key='spconv2', bias=False),
#             nn.BatchNorm1d(channels * 4),
#             nn.ReLU(True)
#         )
#         self.b4 = DoubleConv(channels * 6, channels * 2, indice_key=('subm1', 'subm1'))

#         self.up5 = spconv.SparseSequential(
#             spconv.SparseInverseConv3d(channels * 2, channels * 2, 3, indice_key='spconv1', bias=False),
#             nn.BatchNorm1d(channels * 2),
#             nn.ReLU(True)
#         )
#         self.b5 = DoubleConv(channels * 3, channels, indice_key=('subm0', 'subm0'))

#         self.up6 = spconv.SparseSequential(
#             spconv.SparseInverseConv3d(channels, channels, 3, indice_key='spconv0', bias=True),
#             # nn.BatchNorm1d(channels),
#             # nn.ReLU(True)
#         )

#     def forward(self, input):
#         c1 = self.in_conv(input)
#         c2 = self.f1(c1)
#         c3 = self.f2(c2)
#         c4 = self.f3(c3)
#         c5 = self.f4(c4)
#         c6 = self.f5(c5)

#         up_1 = self.up1(c6)
#         merge1 = spconv.JoinTable()([up_1, c5])
#         cb1 = self.b1(merge1)
#         up_2 = self.up2(cb1)
#         merge2 = spconv.JoinTable()([up_2, c4])
#         cb2 = self.b2(merge2)
#         up_3 = self.up3(cb2)
#         merge3 = spconv.JoinTable()([up_3, c3])
#         cb3 = self.b3(merge3)
#         up_4 = self.up4(cb3)
#         merge4 = spconv.JoinTable()([up_4, c2])
#         cb4 = self.b4(merge4)
#         up_5 = self.up5(cb4)
#         merge5 = spconv.JoinTable()([up_5, c1])
#         cb5 = self.b5(merge5)
#         up_6 = self.up6(cb5)
#         return up_6




# class SpGatUnit(nn.Module):
#     def __init__(self, channels, n_heads, atten_channels=None):
#         super(SpGatUnit, self).__init__()
#         self._n_heads = n_heads
#         self._head_channels = channels // n_heads
#         self._atten_channels = atten_channels if atten_channels is not None else channels // 4
#         self.q_linear = nn.Sequential(
#             nn.Linear(channels, self._atten_channels), nn.BatchNorm1d(self._atten_channels), nn.ReLU(inplace=True)
#         )
#         self.k_linear = nn.Sequential(
#             nn.Linear(channels, self._atten_channels), nn.BatchNorm1d(self._atten_channels), nn.ReLU(inplace=True)
#         )
#         self.v_linear = nn.Sequential(
#             nn.Linear(channels, channels), nn.BatchNorm1d(channels), nn.ReLU(inplace=True)
#         )
#         self.output_linear = nn.Sequential(
#             nn.Linear(channels, channels), nn.BatchNorm1d(channels)
#         )

#     def forward(self, feature, adj_m, mask):
#         # feature=(n, c), mask=(n, n_head, m), points=(n, 3),
#         # adj_m=(n, ch, m)
#         n, c = feature.shape
#         m = mask.shape[3]
#         q = self.q_linear(feature)
#         k = self.k_linear(feature)
#         v = self.v_linear(feature)

#         q = q.view(n, self._n_heads, self._head_channels)
#         k = torch.gather(k, dim=2, index=adj_m)
#         k = k.view(n, self._n_heads, self._head_channels, d, m)
#         v = torch.gather(v, dim=2, index=adj_m)
#         v = v.view(n, self._n_heads, self._head_channels, d, m)

#         atten = q[:, :, :, :, None] * k
#         atten = torch.sum(atten, dim=2)
#         atten = torch.masked_fill(atten, mask, self._eps)
#         atten = torch.softmax(atten, dim=-1)
#         v_halt = atten[:, :, None, :, :] * v
#         v_halt = torch.sum(v_halt, dim=-1)
#         v_halt = v_halt.view(n, -1, d)
#         v_halt = self.output_linear(v_halt)
#         v_halt = v_halt + feature
#         v_halt = F.relu(v_halt)
#         return v_halt
