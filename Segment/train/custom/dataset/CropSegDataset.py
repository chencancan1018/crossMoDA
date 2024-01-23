"""data loader."""

import os
import random
import traceback
from typing import List, Union

import numpy as np
import SimpleITK as sitk
import torch
from starship.umtf.common import build_pipelines
from starship.umtf.common.dataset import DATASETS
from starship.umtf.service.component import CustomDataset, DefaultSampleDataset
from scipy.ndimage import zoom, binary_erosion

@DATASETS.register_module
class CropSegSampleDataset(DefaultSampleDataset):
    def __init__(
            self,
            dst_list_file,
            data_root,
            patch_size,
            sample_frequent,
            pipelines,
    ):
        self._sample_frequent = sample_frequent
        self._patch_size = patch_size
        self._data_file_list = self._load_file_list(data_root, dst_list_file)
        self.draw_idx = 1
        if len(pipelines) == 0:
            self.pipeline = None
        else:
            self.pipeline = build_pipelines(pipelines)

    def _load_file_list(self, data_root, dst_list_file):
        data_file_list = []
        with open(dst_list_file, "r") as f:
            for line in f:
                line = line.strip().split(' ')
                file_name = line[0]
                file_name = os.path.join(data_root, file_name)
                if not os.path.exists(file_name):
                    print(f"{line} not exist")
                    continue
                data_file_list.append(file_name)
        random.shuffle(data_file_list)
        assert len(data_file_list) != 0, "has no avilable file in dst_list_file"
        return data_file_list

    def find_valid_region(self, mask, values, low_margin=[0,0,0], up_margin=[0,0,0]):
        for v in values:
            mask[mask == v] = 100
        nonzero_points = np.argwhere((mask > 20))
        if len(nonzero_points) == 0:
            return None, None
        else:
            v_min = np.min(nonzero_points, axis=0)
            v_max = np.max(nonzero_points, axis=0)
            assert len(v_min) == len(low_margin), f'the length of margin is not equal the mask dims {len(v_min)}!'
            for idx in range(len(v_min)):
                v_min[idx] = max(0, v_min[idx] - low_margin[idx])
                v_max[idx] = min(mask.shape[idx], v_max[idx] + up_margin[idx])
            return v_min, v_max
        
    def random_contrast_bright(self, img, mask, bright_bias=(-0.4, 0.5), bright_weight=(-0.5, 0.5)):
        # rand_mask_b = random.uniform(-0.5, 0) # for round 1
        # rand_mask_b = random.uniform(-0.5, 0.4) # for round 2

        # img contrast, brightness
        img_mean = np.mean(img)
        rand_img_b = np.random.normal(0, 0.1)
        rand_img_w = random.uniform(0.6, 1.5)
        rand_img_multiplier_b = random.uniform(0.6, 1.5) 
        img_out = ((img - img_mean) * rand_img_w + img_mean) * rand_img_multiplier_b + rand_img_b

        # img gamma aug
        img_mean = np.mean(img_out)
        img_sd = np.std(img_out)
        img_min = np.min(img_out)
        img_max = np.max(img_out)
        img_gamma = random.uniform(0.6, 1.5)
        img_out = np.power(((img_out - img_min) / (img_max - img_min + 1e-7)), img_gamma) * (img_max - img_min + 1e-7) + img_min
        img_out = img_out - img_out.mean() + img_mean
        img_out = (img_out / (img_out.std() + 1e-8)) * img_sd # retain stats

        # target vs contrast, brightness, gamma
        vs_pmin, vs_pmax = self.find_valid_region(mask.copy(), [1,2], low_margin=[1,2,2], up_margin=[1,2,2])
        if vs_pmin is not None:
            vs_cube = img_out[vs_pmin[0]:vs_pmax[0], vs_pmin[1]:vs_pmax[1], vs_pmin[2]:vs_pmax[2]].copy()
            vs_mask = mask[vs_pmin[0]:vs_pmax[0], vs_pmin[1]:vs_pmax[1], vs_pmin[2]:vs_pmax[2]].copy()
            vs_mask[vs_mask == 3] = 0
            # 取出vs部分降低cpu负荷
            vs_cube_mean = np.mean(vs_cube * (vs_mask > 0))
            rand_vs_w = random.uniform(0.8, 1.2)
            rand_vs_b = random.uniform(0.8, 1.2) 
            vs_out = ((vs_cube - vs_cube_mean) * rand_vs_w + vs_cube_mean) * rand_vs_b
            # add_vs_b = np.random.normal(0, 0.05)
            # vs_out = vs_out + add_vs_b
            #vs部分 gamma aug
            vs_gamma = random.uniform(0.8, 1.2)
            vs_cube_mean = np.mean(vs_out)
            vs_cube_sd = np.std(vs_out)
            vs_cube_min = np.min(vs_out)
            vs_cube_max = np.max(vs_out)
            vs_out = np.power(((vs_out - vs_cube_min) / (vs_cube_max - vs_cube_min + 1e-7)), vs_gamma) * (vs_cube_max - vs_cube_min + 1e-7) + vs_cube_min
            vs_out = vs_out - vs_out.mean() + vs_cube_mean
            vs_out = (vs_out / (vs_out.std() + 1e-8)) * vs_cube_sd # retain stats
            vs_out = vs_out * ((vs_mask > 0))
            vs_img = np.zeros(img.shape); vs_img[:,:,:] = -1.
            vs_img[vs_pmin[0]:vs_pmax[0], vs_pmin[1]:vs_pmax[1], vs_pmin[2]:vs_pmax[2]] = vs_out
            vs_out = vs_img
        else:
            vs_out = np.zeros(img.shape)

        # cochlea contrast, brightness, gamma
        coc_pmin, coc_pmax = self.find_valid_region(mask.copy(), [3], low_margin=[1,2,2], up_margin=[1,2,2])
        if coc_pmin is not None:
            coc_cube = img_out[coc_pmin[0]:coc_pmax[0], coc_pmin[1]:coc_pmax[1], coc_pmin[2]:coc_pmax[2]].copy()
            coc_mask = mask[coc_pmin[0]:coc_pmax[0], coc_pmin[1]:coc_pmax[1], coc_pmin[2]:coc_pmax[2]].copy()
            coc_mask[coc_mask <= 2] = 0
            # 取出cochlea部分降低cpu负荷
            coc_cube_mean = np.mean(coc_cube * (coc_mask > 0))
            rand_coc_w = random.uniform(0.5, 1.5)
            rand_coc_b = random.uniform(0.3, 2.0) 
            coc_out = ((coc_cube - coc_cube_mean) * rand_coc_w + coc_cube_mean) * rand_coc_b
            # cochlea部分 gamma aug
            coc_gamma = random.uniform(0.5, 2)  
            coc_cube_mean = np.mean(coc_out)
            coc_cube_sd = np.std(coc_out)
            coc_cube_min = np.min(coc_out)
            coc_cube_max = np.max(coc_out)
            coc_out = np.power(((coc_out - coc_cube_min) / (coc_cube_max - coc_cube_min + 1e-7)), coc_gamma) * (coc_cube_max - coc_cube_min + 1e-7) + coc_cube_min
            coc_out = coc_out - coc_out.mean() + coc_cube_mean
            coc_out = (coc_out / (coc_out.std() + 1e-8)) * coc_cube_sd
            coc_out = coc_out * ((coc_mask > 0))
            coc_img = np.zeros(img.shape); coc_img[:, :, :] = -1.
            coc_img[coc_pmin[0]:coc_pmax[0], coc_pmin[1]:coc_pmax[1], coc_pmin[2]:coc_pmax[2]] = coc_out
            coc_out = coc_img
        else:
            coc_out = np.zeros(img.shape)

        nmask = 1 * (mask <= 0.5)
        vsmask = 1 * (mask > 0.5) * (mask < 2.5)
        cmask = 1 * (mask >= 2.5)
        out = img_out * nmask + vs_out * vsmask + coc_out * cmask
        # out = img_out
        return out

    def _load_source_data(self, file_name):
        # print(file_name)
        data = np.load(file_name.split('\n')[0])
        basename = os.path.basename(file_name)
        result = {}
        with torch.no_grad():
            if 'T1' in basename:
                vol = data["fakeT2"]
            elif 'T2' in basename:
                vol = data["realT2"]
            else:
                raise KeyError(f"no T1 or T2 in file: {basename}")
            seg = data["seg"]
            assert (np.array(vol.shape) == np.array(seg.shape)).all()

            # rand y to -1
            depth, w, h = vol.shape
            if random.random() < 0.5:
                start_y = random.randint(0, int(w*1/3))
                end_y = w - start_y
                vol[:, :start_y, :] = -1
                seg[:, :start_y, :] = 0
                vol[:, end_y:, :] = -1
                seg[:, end_y:, :] = 0
            
            if depth < self._patch_size[0]:
                short = self._patch_size[0] - depth
                left_short = int(short // 2)
                right_short = short - left_short
                vol = np.pad(vol, ((left_short, right_short), (0,0), (0,0)), 'constant', constant_values=(-1, -1))
                seg = np.pad(seg, ((left_short, right_short), (0,0), (0,0)), 'constant', constant_values=(0, 0))
                start_z = 0
            else:
                start_z = random.randint(0, (depth-self._patch_size[0]))
            end_z = start_z + self._patch_size[0]
            vol = vol[start_z:end_z, :, :]
            seg = seg[start_z:end_z, :, :]

            # contrast vol and target aug
            vol = self.random_contrast_bright(vol, seg.copy())
            vol = np.clip(vol, -1, 1)

            # resize
            patch_size = np.array(self._patch_size)
            vol_shape = np.array(vol.shape)
            if np.any(vol_shape != patch_size):
                vol = zoom(vol, np.array(patch_size/vol_shape), order=1)
                seg = zoom(seg, np.array(patch_size/vol_shape), order=0)

            vol = torch.from_numpy(vol).float()[None]
            seg = torch.from_numpy(seg).int()[None]

            # other aug
            if self.pipeline:
                vol, seg = self.pipeline(data=(vol, seg))

            result['vol'] = vol.detach()
            result['seg'] = seg.detach()
        del data
        return result

    def _sample_source_data(self, idx, source_data_info):
        try:
            info, _source_data = source_data_info
            return _source_data
        except:
            traceback.print_exc()
            return None
    
    def sample_source_data(self, idx, source_data):
        sample = None
        if idx < self._sample_frequent:
            sample = self._sample_source_data(idx, source_data)
        return sample

    def __getitem__(self, idx):
        source_data = self._load_source_data(self._data_file_list[idx])
        return [None, source_data]

    @property
    def sampled_data_count(self):
        # TODO: sample后数据总数量
        return self.source_data_count * self._sample_frequent

    @property
    def source_data_count(self):
        # TODO: 原始数据总数量
        return len(self._data_file_list)

    def __len__(self):
        return self.source_data_count

    def evaluate(self, results, logger=None):
        res_info = dict()
        res_info['dice'] = np.mean(np.array(results))
        return res_info

    def zscore(self, vol):
        img_mean = vol.mean()
        img_std = vol.std()
        vol = (vol - img_mean) / (img_std + 1e-12)
        return vol


    def evaluate(self, results, logger=None):
        res_info = dict()
        res_info['avg_dice'] = np.mean(np.array([r for r in results]))
        return res_info

    