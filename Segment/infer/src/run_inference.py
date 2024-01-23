import argparse
import glob
import os
import sys

import numpy as np
import SimpleITK as sitk
from skimage import measure

from scipy.ndimage.morphology import binary_erosion, binary_dilation
from tqdm import tqdm
from scipy.ndimage import zoom

from predictor import SegModel, SegPredictor
# from memory_profiler import profile

def compute_dist(mask1, mask2, label=1):
    temp1 = (mask1 == label) * 1
    temp2 = (mask2 == label) * 1
    gap = temp1 - temp2
    dist = (gap != 0).sum() / (temp1.sum() + temp2.sum() + 1)
    return dist

def inference(predictor: SegPredictor, hu_volume):
    pred_array = predictor.forward(hu_volume)
    return pred_array


def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, tmp_img, spacing

def is_flip(direction):
    x_d = direction[0]; y_d = direction[4]; z_d = direction[8]
    if (x_d) < 0:
        x_flip = True
    elif (x_d) > 0:
        x_flip = False
    else:
        raise ValueError(f" wrong x direction {x_d} in sitk img!")
    if (y_d) < 0:
        y_flip = True
    elif (y_d) > 0:
        y_flip = False
    else:
        raise ValueError(f" wrong y direction {y_d} in sitk img!")
    if (z_d) < 0:
        z_flip = True
    elif (z_d) > 0:
        z_flip = False
    else:
        raise ValueError(f" wrong z direction {z_d} in sitk img!")
    return x_flip, y_flip, z_flip

def normalize(vol):
    vol_min = np.percentile(vol, 5)
    vol_max = np.percentile(vol, 95)
    vol = np.clip(vol, vol_min, vol_max)
    vol = (vol - vol_min) / (vol_max - vol_min + 1e-12)
    vol = (vol - 0.5) / 0.5
    return vol

# @profile
def run_infer(x_flip, y_flip, z_flip, hu_volume, predictor_segmask_3d):

    crop_size = np.array([384, 384])
    stride_z = 40

    if x_flip:
        hu_volume = np.ascontiguousarray(np.flip(hu_volume, 2))
    if y_flip:
        hu_volume = np.ascontiguousarray(np.flip(hu_volume, 1))
    if z_flip:
        hu_volume = np.ascontiguousarray(np.flip(hu_volume, 0))
    
    depth, w, h = hu_volume.shape
    # for ukm abnormal data
    if max(w, h) > 350:
        if (w/h < 0.7) or (w/h > 1.3):
            if w > h:
                start_w = int(3 * w / 16); end_w = int(13 * w / 16)
                start_h = 0; end_h = h
            else:
                start_w = 0; end_w = w
                start_h = int(3 * h / 16); end_h = int(13 * h / 16)
        else:
            start_w = int(3 * w / 16); end_w = int(13 * w / 16)
            start_h = int(3 * h / 16); end_h = int(13 * h / 16)
    else:
        start_w = 0; end_w = w
        start_h = 0; end_h = h
    cropped_img = hu_volume[:, start_w:end_w, start_h:end_h]
    size_x, size_y = cropped_img.shape[1:]
    if size_x != size_y:
        left_dis = int(abs(size_x - size_y) // 2)
        right_dis = abs(size_x - size_y) - left_dis
        if size_x > size_y:
            left_short = [0, left_dis]
            right_short = [0, right_dis]
        else:
            left_short = [left_dis, 0]
            right_short = [right_dis, 0]

    # get valid region
    global_seg = np.zeros((depth, w, h))
    cropped_case = np.zeros((depth, crop_size[0], crop_size[1]))
    valid_z_list = []
    for idx in range(depth):
        scan = cropped_img[idx]
        if (scan.max() - scan.min()) > 10:
            # scan = zscore(scan)
            scan = normalize(scan)
            if size_x != size_y:
                scan = np.pad(scan, [[l, r] for l, r in zip(left_short, right_short)], 'constant', constant_values=(-1, -1))
            assert scan.shape[0] == scan.shape[1]
            if np.any(np.array(scan.shape) != crop_size):
                scan = zoom(scan, np.array(np.array(crop_size) / np.array(scan.shape)), order=1)
            cropped_case[idx] = scan
            valid_z_list.append(idx)
    start_d = min(valid_z_list); end_d = max(valid_z_list) + 1
    cropped_case = cropped_case[start_d:end_d]

    # inference
    target_pred = np.zeros(cropped_case.shape)
    size_z = target_pred.shape[0]
    if size_z <= stride_z:
        left_short_z = (stride_z - size_z) // 2
        right_short_z = stride_z - size_z - left_short_z
        input_case = np.pad(cropped_case, ((left_short_z,right_short_z), (0,0), (0,0)), 'constant', constant_values=(-1,-1))
        assert input_case.shape[0] == stride_z
        assert (np.array(input_case.shape[1:]) == crop_size).all()
        pred = inference(predictor_segmask_3d, input_case)
        target_pred = pred[left_short_z:(left_short_z+size_z), :, :]
    else:
        iterations = int(np.ceil(size_z / stride_z))
        start_z_points = [stride_z * i for i in range(iterations)]
        for start_z in start_z_points:
            start_z = min(start_z, (size_z-stride_z))
            input_case = cropped_case[start_z:(start_z+stride_z), :, :]
            assert input_case.shape[0] == stride_z
            assert (np.array(input_case.shape[1:]) == crop_size).all()
            pred = inference(predictor_segmask_3d, input_case)
            target_pred[start_z:(start_z+stride_z), :, :] = pred

    # remove xy padding
    if size_x != size_y:
        scale = np.array(np.array([end_d-start_d, max(size_x, size_y), max(size_x, size_y)]) / np.array(target_pred.shape))
        target_pred = zoom(target_pred, scale, order=0)
        if size_x > size_y:
            target_pred = target_pred[:, :, left_dis:(left_dis + size_y)]
        else:
            target_pred = target_pred[:, left_dis:(left_dis + size_x), :]
    else:
        scale = np.array(np.array([end_d-start_d, end_w-start_w, end_h-start_h]) / np.array(target_pred.shape))
        target_pred = zoom(target_pred, scale, order=0)
    global_seg[start_d:end_d, start_w:end_w, start_h:end_h] = target_pred

    if x_flip:
        global_seg = np.ascontiguousarray(np.flip(global_seg, 2))
    if y_flip:
        global_seg = np.ascontiguousarray(np.flip(global_seg, 1))
    if z_flip:
        global_seg = np.ascontiguousarray(np.flip(global_seg, 0))

    save_seg = global_seg.astype(np.uint8)
    return save_seg

def main(input_path, output_path, gpu, args):

    model_segmask_3d = SegModel(
        model_f=args.model_file,
        network_f=args.network_file,
    )
    predictor_segmask_3d = SegPredictor(
        gpu = gpu,
        model = model_segmask_3d,
    )

    os.makedirs(output_path, exist_ok=True)

    pids = sorted(os.listdir(input_path))
    str1 = [[False, False, False], [True, False, False],[False, True, False],[False, False, True],[True, True, False],[True, False, True],[False, True, True],[True, True, True]]
    bad_case = []
    for pid in tqdm(pids):
        print(pid)
        print('data load ......')
        _, sitk_img, _ = load_nii(os.path.join(input_path, pid))
        hu_volume = sitk.GetArrayFromImage(sitk_img)
        mask0 = run_infer(str1[0][0], str1[0][1], str1[0][2], hu_volume, predictor_segmask_3d)
        mask1 = run_infer(str1[1][0], str1[1][1], str1[1][2], hu_volume, predictor_segmask_3d)
        mask2 = run_infer(str1[2][0], str1[2][1], str1[2][2], hu_volume, predictor_segmask_3d)
        mask3 = run_infer(str1[3][0], str1[3][1], str1[3][2], hu_volume, predictor_segmask_3d)
        mask4 = run_infer(str1[4][0], str1[4][1], str1[4][2], hu_volume, predictor_segmask_3d)
        mask5 = run_infer(str1[5][0], str1[5][1], str1[5][2], hu_volume, predictor_segmask_3d)
        mask6 = run_infer(str1[6][0], str1[6][1], str1[6][2], hu_volume, predictor_segmask_3d)
        mask7 = run_infer(str1[7][0], str1[7][1], str1[7][2], hu_volume, predictor_segmask_3d)

        all_mask_1 = 1 * ((mask0>0) & (mask1>0) & (mask2>0) & (mask3>0) & (mask4>0) & (mask5>0) & (mask6>0) & (mask7>0))
        all_mask_2 = 1 * ((mask0>0) | (mask1>0) | (mask2>0) | (mask3>0) | (mask4>0) | (mask5>0) | (mask6>0) | (mask7>0))
        all_mask_dist = compute_dist(all_mask_1, all_mask_2)
        if all_mask_dist > 0.7:
            bad_case.append(pid)

        if pid in bad_case:
            out = np.zeros(mask0.shape)
            label1_mask = 1 * ((mask0==1) | (mask1==1) | (mask2==1) | (mask3==1) | (mask4==1) | (mask5==1) | (mask6==1) | (mask7==1))
            out[label1_mask == 1] = 1
            label2_mask = 2 * ((mask0==2) | (mask1==2) | (mask2==2) | (mask3==2) | (mask4==2) | (mask5==2) | (mask6==2) | (mask7==2))
            out[label2_mask == 2] = 2
            z_idxes = [i for i in range(out.shape[0]) if label2_mask[i].max() > 0]
            z_mean = np.mean(z_idxes)
            label2_mask = 3 * ((mask0==3) | (mask1==3) | (mask2==3) | (mask3==3) | (mask4==3) | (mask5==3) | (mask6==3) | (mask7==3))
            for i in range(label2_mask.shape[0]):
                if abs(i - z_mean) > 30:
                    label2_mask[i, :, :] = 0
            out[label2_mask == 3] = 3
        else:
            # cascade
            out = np.zeros(mask0.shape)
            # thresh = 0.8
            thresh = 0.7

            label1_mask = 1 * ((mask0==1) & (mask1==1) & (mask2==1) & (mask3==1) & (mask4==1) & (mask5==1) & (mask6==1) & (mask7==1))
            label2_mask = 1 * ((mask0==2) | (mask1==2) | (mask2==2) | (mask3==2) | (mask4==2) | (mask5==2) | (mask6==2) | (mask7==2))
            if label1_mask.sum() > label2_mask.sum():
                label1_first = True
            else:
                label1_first = False 

            label_lesion_mask = ((mask0>=1)*(mask0<=2)*1 + 
                                (mask1>=1)*(mask1<=2)*1 + 
                                (mask2>=1)*(mask2<=2)*1 + 
                                (mask3>=1)*(mask3<=2)*1 + 
                                (mask4>=1)*(mask4<=2)*1 + 
                                (mask5>=1)*(mask5<=2)*1 + 
                                (mask6>=1)*(mask6<=2)*1 + 
                                (mask7>=1)*(mask7<=2)*1) / 8
            label_lesion_mask[label_lesion_mask > thresh] = 1
            label_lesion_mask[label_lesion_mask <= thresh] = 0
            
            label1_mask = ((mask0==1)*1 + (mask1==1)*1 + (mask2==1)*1 + (mask3==1)*1 + (mask4==1)*1 + (mask5==1)*1 + (mask6==1)*1 + (mask7==1)*1) / 8
            label1_mask[label1_mask > thresh] = 1
            label1_mask[label1_mask <= thresh] = 0

            label2_mask = ((mask0==2)*1 + (mask1==2)*1 + (mask2==2)*1 + (mask3==2)*1 + (mask4==2)*1 + (mask5==2)*1 + (mask6==2)*1 + (mask7==2)*1) / 8
            label2_mask[label2_mask > thresh] = 1
            label2_mask[label2_mask <= thresh] = 0
            if label2_mask.max() > 0:
                label2_labeled = measure.label(label2_mask)
                label2_regions = measure.regionprops(label2_labeled)
                label2_region = sorted(label2_regions, key=lambda r:r.area, reverse=True)[0]
                label2_labeled[label2_labeled != label2_region.label] = 0
                label2_mask = 2 * label2_mask * (label2_labeled > 0)

            if label1_first:
                out[label1_mask == 1] = 1
                out[label2_mask == 2] = 2
                hole_mask = label_lesion_mask - ((label1_mask + label2_mask) > 0) * label_lesion_mask
                out[hole_mask > 0] = 2
            else:
                out[label2_mask == 2] = 2
                out[label1_mask == 1] = 1
                hole_mask = label_lesion_mask - ((label1_mask + label2_mask) > 0) * label_lesion_mask
                out[hole_mask > 0] = 1
            # print(pid, np.unique(hole_mask))
            # print(pid, label1_mask.sum(), label2_mask.sum(), hole_mask.sum())

            label3_mask = ((mask0==3)*1 + (mask1==3)*1 + (mask2==3)*1 + (mask3==3)*1 + (mask4==3)*1 + (mask5==3)*1 + (mask6==3)*1 + (mask7==3)*1) / 8
            label3_mask[label3_mask > 0.5] = 3
            label3_mask[label3_mask <= 0.5] = 0
            # label3_mask = 3 * ((mask0==3) & (mask1==3) & (mask2==3) & (mask3==3) & (mask4==3) & (mask5==3) & (mask6==3) & (mask7==3))
            if label3_mask.sum() == 0:
                # print(pid, f"cochlea seg is empty!!")
                label3_mask = 1 * ((mask0==3) | (mask1==3) | (mask2==3) | (mask3==3) | (mask4==3) | (mask5==3) | (mask6==3) | (mask7==3))
                label3_labeled = measure.label(label3_mask)
                label3_regions = measure.regionprops(label3_labeled)
                label3_regions = sorted(label3_regions, key=lambda r:r.area, reverse=True)
                label3_regions = [r for r in label3_regions if r.area > 50]
                label3_mask[label1_mask != 0] = 0
                for r in label3_regions:
                    label3_mask[label3_labeled == r.label] = 3
            out[label3_mask == 3] = 3
            out = out.astype(np.uint8)

            # max connected region
            temp = out.copy()
            temp = (temp > 0) * (temp < 3)
            labeled = measure.label(temp)
            regions = measure.regionprops(labeled)
            if len(regions) > 0:
                if len(regions) > 1:
                    print(pid, f'lesion num:{len(regions)}, {[r.area for r in regions]}')
                    labels_num = [len(np.unique((labeled == r.label) * out)) - 1 for r in regions]
                    if max(labels_num) == 1:
                        regions = sorted(regions, key=lambda r: r.area, reverse=True)
                        max_region = regions[0]
                        temp = (labeled == max_region.label) * 1
                    elif max(labels_num) == 2:
                        regions = [r for r in regions if len(np.unique((labeled == r.label) * out)) == 3 ]
                        regions = sorted(regions, key=lambda r: r.area, reverse=True)
                        max_region = regions[0]
                        temp = (labeled == max_region.label) * 1
                else:
                    temp = (labeled> 0) * 1
            else:
                print(pid, 'lesion num: 0')
                
            temp[out == 3] = 1
            out = out * temp
            out = out.astype(np.uint8)
    
        save_seg = out
        print('Check output classes: ', np.unique(save_seg))
        segments_itk = sitk.GetImageFromArray(save_seg)
        segments_itk.CopyInformation(sitk_img)
        save_name = pid.replace('_T2', '')
        sitk.WriteImage(segments_itk, os.path.join(output_path, f'{save_name}'))

def parse_args():
    parser = argparse.ArgumentParser(description='Test for abdomen_seg_mask3d')

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--input_path', default='/input/', type=str)
    parser.add_argument('--output_path', default='/output/', type=str)
    # parser.add_argument('--input_path', default='./input/', type=str)
    # parser.add_argument('--output_path', default='./output/', type=str)

    parser.add_argument('--model_file', default='/src/data/model/epoch_120.pth', type=str,)
    parser.add_argument('--network_file', default='/src/data/model/config_cube.py', type=str,)
    # parser.add_argument('--model_file', default='./data/model/epoch_120.pth', type=str,)
    # parser.add_argument('--network_file', default='./data/model/config_cube.py', type=str,)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    fine_seg = False
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        gpu=args.gpu,
        args=args,
    )
