import math
import random

import torch
import torch.nn as nn
import torchvision
import numpy as np

from starship.umtf.common.dataset import PIPELINES

from .transforms.croppad.array import RandScaleCrop
from .transforms.croppad.dictionary import RandScaleCropD
from .transforms.intensity.array import RandGaussianSharpen, RandShiftIntensity, RandScaleIntensity, ScaleIntensityRange, RandAdjustContrast
from .transforms.intensity.dictionary import RandGaussianSharpenD, RandShiftIntensityD, RandScaleIntensityD, ScaleIntensityRangeD, RandAdjustContrastD
from .transforms.smooth_field.array import RandSmoothDeform
from .transforms.smooth_field.dictionary import RandSmoothDeformDict
from .transforms.spatial.array import RandRotate90, RandFlip, RandAffine, Rand3DElastic
from .transforms.spatial.dictionary import RandAffineD, Rand3DElasticD
from .transforms.utility.array import ToTensor
from .transforms.utility.dictionary import ToTensorD
from .transforms.spatial.dictionary import RandRotate90d, RandFlipd


@PIPELINES.register_module
class MonaiAffineTransform():
    
    def __init__(self, aug_parameters: dict):

        self.patch_size = aug_parameters.setdefault("patch_size", (128, 128, 128))
        self.roi_scale = aug_parameters.setdefault("roi_scale", (1.0, 1.0, 1.0)) 
        self.max_roi_scale = aug_parameters.setdefault('max_roi_scale', (1.0, 1.0, 1.0)) 
        self.rotate_x = aug_parameters.setdefault('rot_range_x', (-np.pi/9, np.pi/9))
        self.rotate_y = aug_parameters.setdefault('rot_range_y', (-np.pi/9, np.pi/9)) 
        self.rotate_z = aug_parameters.setdefault('rot_range_z', (-np.pi/9, np.pi/9)) 
        self.rotate_90 = aug_parameters.setdefault('rot_90', False)
        self.flip = aug_parameters.setdefault("flip", False)
        self.prob = aug_parameters.setdefault('prob', 0.5) 
        self.bright_bias = aug_parameters.setdefault('bright_bias', (-0.2, 0.2)) 
        self.bright_w = aug_parameters.setdefault('bright_weight', (-0.2, 0.2)) 
        self.translate_x = aug_parameters.setdefault('translate_x', (-5.0, 5.0))
        self.translate_y = aug_parameters.setdefault('translate_y', (-5.0, 5.0))
        self.translate_z = aug_parameters.setdefault('translate_z', (-5.0, 5.0))
        self.scale_x = aug_parameters.setdefault('scale_x', (-0.1, 0.1)) 
        self.scale_y = aug_parameters.setdefault('scale_y', (-0.1, 0.1)) 
        self.scale_z = aug_parameters.setdefault('scale_z', (-0.1, 0.1)) 
        self.elastic_sigma_range =  aug_parameters.setdefault('elastic_sigma_range', (3, 5))
        self.elastic_magnitude_range = aug_parameters.setdefault('elastic_magnitude_range', (100, 200))

        augmentations = list()

        # random crop
        if self.roi_scale != self.max_roi_scale:
            rand_crop = RandScaleCrop(roi_scale=self.roi_scale, max_roi_scale=self.max_roi_scale)
            augmentations.append(rand_crop)

        if self.rotate_90:
            rand_rotate90 = RandRotate90(prob=self.prob, max_k=3)
            augmentations.append(rand_rotate90)
        
        if self.flip:
            rand_flip_z = RandFlip(spatial_axis=[0], prob=self.prob) 
            rand_flip_y = RandFlip(spatial_axis=[1], prob=self.prob)
            rand_flip_x = RandFlip(spatial_axis=[2], prob=self.prob)
            augmentations.append(rand_flip_x)
            augmentations.append(rand_flip_y)
            augmentations.append(rand_flip_z)

        # gaussian blur
        gaussian_blur = RandGaussianSharpen()
        augmentations.append(gaussian_blur)

        # brightness 
        """
        offset: 0.2, 图像像素偏差值范围为(-0.2, 0.2) 
        factors: 0.2, 图像像素权重值范围为(1-0.2, 1+0.2) 
        """
        intensity_shift = RandShiftIntensity(offsets=self.bright_bias, prob=self.prob)
        intensity_scale = RandScaleIntensity(factors=self.bright_w, prob=self.prob)
        augmentations.append(intensity_shift)
        augmentations.append(intensity_scale)

        # affine transform
        """"
        prob: 0.5, 依概率0.5进行仿射变换
        rotate_range: (np.pi/9, np.pi/9, np.pi/9), 图像旋转, x,y,z轴的旋转范围分别为(-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9)
        shear_range: None, 图像剪切, 
        translate_range: (5, 5, 5), 图像平移, x,y,z轴的平移范围分别为(-5, 5), (-5, 5), (-5, 5)
        scale_range: (0.1, 0.1, 0.1), 图像缩放, x,y,z轴的缩放范围为(1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1)
        spatial_size: patch_size, 输出给定尺寸的image patch
        mode: `'bilinear'` (for image), `'nearest'` (for label)
        padding_mode: `'border'`, `'reflection'`, `'zeros'`
        """
        rand_affine = RandAffine(
            prob=self.prob,
            rotate_range= (self.rotate_x, self.rotate_y, self.rotate_z), # 旋转
            shear_range=None,
            translate_range=(self.translate_x, self.translate_y, self.translate_z), # 平移
            scale_range=(self.scale_x, self.scale_y, self.scale_z),  # 缩放
            spatial_size=self.patch_size,  
            mode='bilinear', #'nearest' for label
            padding_mode='border',
        )
        augmentations.append(rand_affine)

        # clip
        clip_0_1 = ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True)
        augmentations.append(clip_0_1)
        
        to_tensor = ToTensor(dtype=torch.float32)
        augmentations.append(to_tensor)

        
        self.transforms = torchvision.transforms.Compose(augmentations)
    
    def __call__(self, x):
        return self.transforms(x)

@PIPELINES.register_module
class MonaiAffineDictTransform():
    
    def __init__(self, aug_parameters: dict):

        self.patch_size = aug_parameters.setdefault("patch_size", (128, 128, 128))
        self.roi_scale = aug_parameters.setdefault("roi_scale", (1.0, 1.0, 1.0)) 
        self.max_roi_scale = aug_parameters.setdefault('max_roi_scale', (1.0, 1.0, 1.0)) 
        self.rotate_x = aug_parameters.setdefault('rot_range_x', (-np.pi/9, np.pi/9))
        self.rotate_y = aug_parameters.setdefault('rot_range_y', (-np.pi/9, np.pi/9)) 
        self.rotate_z = aug_parameters.setdefault('rot_range_z', (-np.pi/9, np.pi/9)) 
        self.rotate_90 = aug_parameters.setdefault('rot_90', False)
        self.flip = aug_parameters.setdefault("flip", False)
        self.prob = aug_parameters.setdefault('prob', 0.5) 
        self.bright_bias = aug_parameters.setdefault('bright_bias', (-0.2, 0.2)) 
        self.bright_w = aug_parameters.setdefault('bright_weight', (-0.2, 0.2)) 
        self.translate_x = aug_parameters.setdefault('translate_x', (-5.0, 5.0))
        self.translate_y = aug_parameters.setdefault('translate_y', (-5.0, 5.0))
        self.translate_z = aug_parameters.setdefault('translate_z', (-5.0, 5.0))
        self.scale_x = aug_parameters.setdefault('scale_x', (-0.1, 0.1)) 
        self.scale_y = aug_parameters.setdefault('scale_y', (-0.1, 0.1)) 
        self.scale_z = aug_parameters.setdefault('scale_z', (-0.1, 0.1)) 
        self.elastic_sigma_range =  aug_parameters.setdefault('elastic_sigma_range', (3, 5))
        self.elastic_magnitude_range = aug_parameters.setdefault('elastic_magnitude_range', (100, 200))

        aug_dict = list()
        aug_img = list()

        # random crop
        if self.roi_scale != self.max_roi_scale:
            rand_crop = RandScaleCropD(keys=["image", "label"], roi_scale=self.roi_scale, max_roi_scale=self.max_roi_scale)
            aug_dict.append(rand_crop)

        if self.rotate_90:
            rand_rotate90d = RandRotate90d(keys=["image", "label"], prob=self.prob, max_k=3)
            aug_dict.append(rand_rotate90d)
        
        if self.flip:
            rand_flipd_z = RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=self.prob) 
            rand_flipd_y = RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=self.prob)
            rand_flipd_x = RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=self.prob)
            aug_dict.append(rand_flipd_x)
            aug_dict.append(rand_flipd_y)
            aug_dict.append(rand_flipd_z)

        #seg affine transform
        """"
        keys: ["image", "label"], data字典中的keys
        prob: 0.5, 依概率0.5进行仿射变换
        rotate_range: (np.pi/9, np.pi/9, np.pi/9), 图像旋转, x,y,z轴的旋转范围分别为(-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9)
        shear_range: None, 图像剪切, 
        translate_range: (5, 5, 5), 图像平移, x,y,z轴的平移范围分别为(-5, 5), (-5, 5), (-5, 5)
        scale_range: (0.1, 0.1, 0.1), 图像缩放, x,y,z轴的缩放范围为(1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1)
        spatial_size: patch_size, 输出给定尺寸的image patch
        mode: `'bilinear'` (for image), `'nearest'` (for label)
        padding_mode: `'border'`, `'reflection'`, `'zeros'`
        """
        rand_affine = RandAffineD(
            keys=["image", "label"],
            spatial_size=self.patch_size,
            prob=self.prob,
            rotate_range= (self.rotate_x, self.rotate_y, self.rotate_z), # 旋转
            shear_range=None,
            translate_range=(self.translate_x, self.translate_y, self.translate_z), # 平移
            scale_range=(self.scale_x, self.scale_y, self.scale_z),  # 缩放
            mode=("bilinear", "nearest"),
            padding_mode='border',
        )
        aug_dict.append(rand_affine)


        # gaussian blur
        gaussian_blur = RandGaussianSharpen()
        aug_img.append(gaussian_blur)

        # brightness 
        """
        offset: 0.2, 图像像素偏差值范围为(-0.2, 0.2) 
        factors: 0.2, 图像像素权重值范围为(1-0.2, 1+0.2) 
        """
        intensity_shift = RandShiftIntensity(offsets=self.bright_bias, prob=self.prob)
        intensity_scale = RandScaleIntensity(factors=self.bright_w, prob=self.prob)
        aug_img.append(intensity_shift)
        aug_img.append(intensity_scale)

        # clip
        clip_0_1 = ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True)
        aug_img.append(clip_0_1)
        
        array_to_tensor = ToTensor(dtype=torch.float32)
        dict_to_tensor = ToTensorD(keys=["image", "label"], dtype=torch.float32)
        aug_img.append(array_to_tensor)
        aug_dict.append(dict_to_tensor)


        self.aug_dict = torchvision.transforms.Compose(aug_dict)
        self.aug_img = torchvision.transforms.Compose(aug_img)
    
    def __call__(self, data):
        img, mask = data
        data_dict = {'image': img, 'label': mask}
        data_dict = self.aug_dict(data_dict)

        img = data_dict["image"]
        mask = data_dict['label']
        img = self.aug_img(img)
        data = img, mask
        return data

@PIPELINES.register_module
class MonaiElasticTransform():
    
    def __init__(self, aug_parameters: dict):

        self.patch_size = aug_parameters.setdefault("patch_size", (128, 128, 128))
        self.roi_scale = aug_parameters.setdefault("roi_scale", (1.0, 1.0, 1.0)) 
        self.max_roi_scale = aug_parameters.setdefault('max_roi_scale', (1.0, 1.0, 1.0)) 
        self.rotate_x = aug_parameters.setdefault('rot_range_x', (-np.pi/9, np.pi/9))
        self.rotate_y = aug_parameters.setdefault('rot_range_y', (-np.pi/9, np.pi/9)) 
        self.rotate_z = aug_parameters.setdefault('rot_range_z', (-np.pi/9, np.pi/9)) 
        self.rotate_90 = aug_parameters.setdefault('rot_90', False)
        self.flip = aug_parameters.setdefault("flip", False)
        self.prob = aug_parameters.setdefault('prob', 0.5) 
        self.bright_bias = aug_parameters.setdefault('bright_bias', (-0.2, 0.2)) 
        self.bright_w = aug_parameters.setdefault('bright_weight', (-0.2, 0.2)) 
        self.translate_x = aug_parameters.setdefault('translate_x', (-5.0, 5.0))
        self.translate_y = aug_parameters.setdefault('translate_y', (-5.0, 5.0))
        self.translate_z = aug_parameters.setdefault('translate_z', (-5.0, 5.0))
        self.scale_x = aug_parameters.setdefault('scale_x', (-0.1, 0.1)) 
        self.scale_y = aug_parameters.setdefault('scale_y', (-0.1, 0.1)) 
        self.scale_z = aug_parameters.setdefault('scale_z', (-0.1, 0.1)) 
        self.elastic_sigma_range =  aug_parameters.setdefault('elastic_sigma_range', (3, 5))
        self.elastic_magnitude_range = aug_parameters.setdefault('elastic_magnitude_range', (100, 200))

        augmentations = list()

        # random crop
        if self.roi_scale != self.max_roi_scale:
            rand_crop = RandScaleCrop(roi_scale=self.roi_scale, max_roi_scale=self.max_roi_scale)
            augmentations.append(rand_crop)

        if self.rotate_90:
            rand_rotate90 = RandRotate90(prob=self.prob, max_k=3)
            augmentations.append(rand_rotate90)
        
        if self.flip:
            rand_flip_z = RandFlip(spatial_axis=[0], prob=self.prob) 
            rand_flip_y = RandFlip(spatial_axis=[1], prob=self.prob)
            rand_flip_x = RandFlip(spatial_axis=[2], prob=self.prob)
            augmentations.append(rand_flip_x)
            augmentations.append(rand_flip_y)
            augmentations.append(rand_flip_z)

        # gaussian blur
        gaussian_blur = RandGaussianSharpen()
        augmentations.append(gaussian_blur)

        # brightness 
        """
        offset: 0.2, 图像像素偏差值范围为(-0.2, 0.2) 
        factors: 0.2, 图像像素权重值范围为(1-0.2, 1+0.2) 
        """
        intensity_shift = RandShiftIntensity(offsets=self.bright_bias, prob=self.prob)
        intensity_scale = RandScaleIntensity(factors=self.bright_w, prob=self.prob)
        augmentations.append(intensity_shift)
        augmentations.append(intensity_scale)
        
        # elastic deformation
        """
        sigma_range: (3, 5), 弹性形变高斯核的方差范围
        magnitude_range: (100, 200), 弹性形变的变化幅度, 若值超过500,图像逐渐虚化
        prob: 0.5, 依概率0.5进行弹性形变
        rotate_range: (np.pi/9, np.pi/9, np.pi/9), 图像分别以x,y,z轴为中心进行,范围分别为(-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9)
        shear_range: None, 图像剪切, 
        translate_range: (5, 5, 5), 图像平移, 中心点的x,y,z轴平移范围分别为(-5, 5), (-5, 5), (-5, 5)
        scale_range: (0.1, 0.1, 0.1), 图像缩放, 图像x,y,z轴的缩放范围为(1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1)
        spatial_size: patch_size, 输出给定尺寸的image patch
        mode: `'bilinear'` (for image), `'nearest'` (for label)
        padding_mode: `'border'`, `'reflection'`, `'zeros'`
        """
        rand_elastic = Rand3DElastic(
            sigma_range=self.elastic_sigma_range,
            magnitude_range=self.elastic_magnitude_range,
            prob=self.prob,
            rotate_range=(self.rotate_x, self.rotate_y, self.rotate_z),
            shear_range=None,
            translate_range=(self.translate_x, self.translate_y, self.translate_z),
            scale_range=(self.scale_x, self.scale_y, self.scale_z),
            spatial_size=self.patch_size,
            mode='bilinear', #'nearest' for label,
            padding_mode='border',
        )
        augmentations.append(rand_elastic)

        # clip
        clip_0_1 = ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True)
        augmentations.append(clip_0_1)
        
        to_tensor = ToTensor(dtype=torch.float32)
        augmentations.append(to_tensor)

        
        self.transforms = torchvision.transforms.Compose(augmentations)
    
    def __call__(self, x):
        return self.transforms(x)


@PIPELINES.register_module
class MonaiElasticDictTransform():
    
    def __init__(self, aug_parameters: dict):

        self.patch_size = aug_parameters.setdefault("patch_size", (128, 128, 128))
        self.roi_scale = aug_parameters.setdefault("roi_scale", (1.0, 1.0, 1.0)) 
        self.max_roi_scale = aug_parameters.setdefault('max_roi_scale', (1.0, 1.0, 1.0)) 
        self.rotate_x = aug_parameters.setdefault('rot_range_x', (-np.pi/9, np.pi/9))
        self.rotate_y = aug_parameters.setdefault('rot_range_y', (-np.pi/9, np.pi/9)) 
        self.rotate_z = aug_parameters.setdefault('rot_range_z', (-np.pi/9, np.pi/9)) 
        self.rotate_90 = aug_parameters.setdefault('rot_90', False)
        self.flip = aug_parameters.setdefault("flip", False)
        self.prob = aug_parameters.setdefault('prob', 0.5) 
        self.bright_bias = aug_parameters.setdefault('bright_bias', (-0.2, 0.2)) 
        self.bright_w = aug_parameters.setdefault('bright_weight', (-0.2, 0.2)) 
        self.translate_x = aug_parameters.setdefault('translate_x', (-5.0, 5.0))
        self.translate_y = aug_parameters.setdefault('translate_y', (-5.0, 5.0))
        self.translate_z = aug_parameters.setdefault('translate_z', (-5.0, 5.0))
        self.scale_x = aug_parameters.setdefault('scale_x', (-0.1, 0.1)) 
        self.scale_y = aug_parameters.setdefault('scale_y', (-0.1, 0.1)) 
        self.scale_z = aug_parameters.setdefault('scale_z', (-0.1, 0.1)) 
        self.elastic_sigma_range =  aug_parameters.setdefault('elastic_sigma_range', (3, 5))
        self.elastic_magnitude_range = aug_parameters.setdefault('elastic_magnitude_range', (100, 200))

        aug_dict = list()
        aug_img = list()

        # random crop
        if self.roi_scale != self.max_roi_scale:
            rand_crop = RandScaleCropD(keys=["image", "label"], roi_scale=self.roi_scale, max_roi_scale=self.max_roi_scale)
            aug_dict.append(rand_crop)
        
        if self.rotate_90:
            rand_rotate90d = RandRotate90d(keys=["image", "label"], prob=self.prob, max_k=3)
            aug_dict.append(rand_rotate90d)
        
        if self.flip:
            rand_flipd_z = RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=self.prob) 
            rand_flipd_y = RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=self.prob)
            rand_flipd_x = RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=self.prob)
            aug_dict.append(rand_flipd_x)
            aug_dict.append(rand_flipd_y)
            aug_dict.append(rand_flipd_z)

        #seg elastic deformation
        """
        keys: ["image", "label"], data字典中的keys
        sigma_range: (3, 5), 弹性形变高斯核的方差范围
        magnitude_range: (100, 200), 弹性形变的变化幅度, 若值超过500,图像逐渐虚化
        prob: 0.5, 依概率0.5进行弹性形变
        rotate_range: (np.pi/9, np.pi/9, np.pi/9), 图像分别以x,y,z轴为中心进行,范围分别为(-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9), (-np.pi/9, np.pi/9)
        shear_range: None, 图像剪切, 
        translate_range: (5, 5, 5), 图像平移, 中心点的x,y,z轴平移范围分别为(-5, 5), (-5, 5), (-5, 5)
        scale_range: (0.1, 0.1, 0.1), 图像缩放, 图像x,y,z轴的缩放范围为(1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1), (1.0-0.1, 1.0+0.1)
        spatial_size: patch_size, 输出给定尺寸的image patch
        mode: `'bilinear'` (for image), `'nearest'` (for label)
        padding_mode: `'border'`, `'reflection'`, `'zeros'`
        """
        rand_elastic = Rand3DElasticD(
            keys=["image", "label"],
            sigma_range=self.elastic_sigma_range,
            magnitude_range=self.elastic_magnitude_range,
            prob=self.prob,
            rotate_range=(self.rotate_x, self.rotate_y, self.rotate_z),
            shear_range=None,
            translate_range=(self.translate_x, self.translate_y, self.translate_z),
            scale_range=(self.scale_x, self.scale_y, self.scale_z),
            spatial_size=self.patch_size,
            mode=("bilinear", "nearest"), #'nearest' for label,
            padding_mode='border',
        )
        aug_dict.append(rand_elastic)

        # gaussian blur
        gaussian_blur = RandGaussianSharpen()
        aug_img.append(gaussian_blur)

        # # brightness 
        # """
        # offset: 0.2, 图像像素偏差值范围为(-0.2, 0.2) 
        # factors: 0.2, 图像像素权重值范围为(1-0.2, 1+0.2) 
        # """
        # intensity_shift = RandShiftIntensity(offsets=self.bright_bias, prob=self.prob)
        # intensity_scale = RandScaleIntensity(factors=self.bright_w, prob=self.prob)
        # aug_img.append(intensity_shift)
        # aug_img.append(intensity_scale)

        # # clip
        # clip_0_1 = ScaleIntensityRange(a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True)
        # aug_img.append(clip_0_1)
        
        array_to_tensor = ToTensor(dtype=torch.float32)
        dict_to_tensor = ToTensorD(keys=["image", "label"], dtype=torch.float32)
        aug_img.append(array_to_tensor)
        aug_dict.append(dict_to_tensor)

        self.aug_dict = torchvision.transforms.Compose(aug_dict)
        self.aug_img = torchvision.transforms.Compose(aug_img)
    
    def __call__(self, data):
        img, mask = data
        data_dict = {'image': img, 'label': mask}
        data_dict = self.aug_dict(data_dict)

        img = data_dict["image"]
        mask = data_dict['label']
        img = self.aug_img(img)
        data = img, mask
        return data

if __name__ == "__main__":
    transforms = MonaiElasticDictTransform(aug_parameters={})
    print(transforms)
    img = np.random.randn(1,128,128,128)
    mask = np.zeros(img.shape, dtype=np.uint8)
    mask[:, 30:60, 30:60, 30:60] = 1
    print(img.shape, mask.shape)
    aug_img, aug_mask = transforms(img, mask)
    print(aug_img.shape, aug_mask.shape)
    print(torch.max(aug_img), torch.min(aug_img))
    
