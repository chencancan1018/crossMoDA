"""生成模型输入数据."""

import argparse
import glob
import json
import os
import random
import sys
import traceback

import numpy as np
import SimpleITK as sitk
import threadpool
import threading
from queue import Queue
from tqdm import tqdm
from scipy.ndimage import zoom
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--tgt_path', type=str, default='/home/tx-deepocean/Data/Project/crossMoDA/IARSeg/example/data/coarse_0605_epoch190/round1_posted')
    # parser.add_argument('--out_path', type=str, default='./checkpoints/predata_fakeT2_round1')
    parser.add_argument('--tgt_path', type=str, default='/home/tx-deepocean/Data/Project/crossMoDA/IARSeg/example/data/coarse_0607_epoch160_round2/round2_posted')
    parser.add_argument('--out_path', type=str, default='./checkpoints/predata_fakeT2_round2')
    parser.add_argument('--src_path', type=str, default='/home/tx-deepocean/Data/Project/crossMoDA/data/crossmoda23_training/TrainingTarget')

    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    vol = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing = spacing[::-1]
    return vol, img, spacing


def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    origin_coord = tmp_img.GetOrigin()
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, tmp_img, spacing

def normalize(vol):
    vol_min = np.percentile(vol, 5)
    vol_max = np.percentile(vol, 95)
    vol = np.clip(vol, vol_min, vol_max)
    vol = (vol - vol_min) / (vol_max - vol_min + 1e-12)
    vol = (vol - 0.5) / 0.5
    return vol

def is_flip(direction):
    x_d = direction[0]; y_d = direction[4]; z_d = direction[8]
    if x_d < 0:
        x_flip = True
    elif x_d > 0:
        x_flip = False
    else:
        raise ValueError(f" wrong x direction {x_d} in sitk img!")
    if y_d < 0:
        y_flip = True
    elif y_d > 0:
        y_flip = False
    else:
        raise ValueError(f" wrong y direction {y_d} in sitk img!")
    if z_d < 0:
        z_flip = True
    elif z_d > 0:
        z_flip = False
    else:
        raise ValueError(f" wrong z direction {z_d} in sitk img!")
    return x_flip, y_flip, z_flip

patch_size = [64, 256, 256] # for coarse- and fine-seg

def gen_data(info):
    
    try:
        pid, vol_file, out_dir, seg_file, sitk_lock = info
        save_file = os.path.join(out_dir, f'{pid}_T2.npz')
        sitk_lock.acquire()
        vol, sitk_img, spacing_vol = load_nii(vol_file)
        sitk_lock.release()

        if  not os.path.exists(seg_file):
            return None
        seg, _, _ = load_nii(seg_file)
        seg = seg.astype(np.uint8)   

        vol_shape = np.array(vol.shape)
        seg_shape = np.array(seg.shape)
        if np.any(vol_shape != seg_shape):
            print('pid vol shape != seg shape: ', pid)
            return None

        x_flip, y_flip, z_flip = is_flip(sitk_img.GetDirection())
        print(pid, sitk_img.GetSize(), sitk_img.GetDirection(), x_flip, y_flip, z_flip)
        if x_flip:
            vol = np.ascontiguousarray(np.flip(vol, 2))
            seg = np.ascontiguousarray(np.flip(seg, 2)).astype(np.uint8)
        if y_flip:
            vol = np.ascontiguousarray(np.flip(vol, 1))
            seg = np.ascontiguousarray(np.flip(seg, 1)).astype(np.uint8)
        if z_flip:
            vol = np.ascontiguousarray(np.flip(vol, 0))
            seg = np.ascontiguousarray(np.flip(seg, 0)).astype(np.uint8)  

        depth, w, h = vol.shape
        start_w = int(3 * w / 16); end_w = int(13 * w / 16)
        start_h = int(3 * h / 16); end_h = int(13 * h / 16)
        if w != h:
                if w > h:
                    start_w = int(3 * w / 16); end_w = int(13 * w / 16)
                    start_h = 0; end_h = h
                else:
                    start_w = 0; end_w = w
                    start_h = int(3 * h / 16); end_h = int(13 * h / 16)
        else:
            start_w = int(3 * w / 16); end_w = int(13 * w / 16)
            start_h = int(3 * h / 16); end_h = int(13 * h / 16)

        cropped_img = vol[:, start_w:end_w, start_h:end_h]
        cropped_label = seg[:, start_w:end_w, start_h:end_h]
        assert (np.array(cropped_img.shape) == np.array(cropped_label.shape)).all()

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
        
        
        cropped_save_img = np.zeros((depth, patch_size[1], patch_size[2]))
        for idx in range(depth):
            scan = cropped_img[idx]
            if (scan.max() - scan.min()) > 10:
                # scan = zscore(scan)
                scan = normalize(scan)
                if size_x != size_y:
                    scan = np.pad(scan, [[l, r] for l, r in zip(left_short, right_short)], 'constant', constant_values=(-1, -1))
                    assert scan.shape[0] == scan.shape[1]
                if np.any(np.array(scan.shape) != np.array(patch_size[1:])):
                    scan = zoom(scan, np.array(np.array(patch_size[1:]) / np.array(scan.shape)), order=1)
                cropped_save_img[idx] = scan
        
        assert (np.array(cropped_label.shape) == np.array([depth, size_x, size_y])).all()
        if size_x != size_y:
            left_short = [0] + left_short; right_short = [0] + right_short # 2D to 3D
            cropped_label = np.pad(cropped_label, [[l, r] for l, r in zip(left_short, right_short)], 'constant', constant_values=(0, 0, 0))
        assert (np.array(cropped_label.shape) == np.array([depth, max(size_x, size_y), max(size_x, size_y)])).all()
        cropped_label = zoom(cropped_label, np.array(np.array((depth, patch_size[1], patch_size[2])) / np.array(cropped_label.shape)), order=0)

        np.savez_compressed(
            save_file,
            realT2=cropped_save_img,
            seg=cropped_label,
            src_spacing=np.array(spacing_vol),
        )
        print(f'{pid} successed')
        return save_file
    except:
        traceback.print_exc()
        sitk_lock.release()
        print(f'{pid} failed')
        return None

def write_list(request, result):
    write_queue.put(result)

def list_save(data_list, out):
    with open(out, 'w') as f:
        for data in data_list:
            f.writelines(data + '\r\n')

def gen_lst(out_dir, fold_num=5):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=fold_num)
    data_list = sorted(os.listdir(out_dir))
    random.shuffle(data_list)
    start = 0
    for train_idx, test_idx in kf.split(data_list):
        train_lst = sorted([data_list[i] for i in train_idx])
        test_lst = sorted([data_list[i] for i in test_idx])
        list_save(train_lst, os.path.join(out_dir, 'train_' + str(start) + '.lst'))
        list_save(test_lst, os.path.join(out_dir, 'val_' + str(start) + '.lst'))
        start += 1

if __name__ == '__main__':
    sitk_lock = threading.RLock()
    write_queue = Queue()
    args = parse_args()
    source_dir = args.src_path
    target_dir = args.tgt_path
    out_dir = args.out_path

    os.makedirs(out_dir, exist_ok=True)
    data_lst = []
    pids = sorted([p[:-7] for p in sorted(os.listdir(target_dir))])
    for pid in pids:
        vol_file = os.path.join(source_dir, pid+'_T2.nii.gz')
        seg_file = os.path.join(target_dir, pid+'.nii.gz')
        info = [pid.replace('.nii.gz', ''), vol_file, out_dir, seg_file, sitk_lock]
        data_lst.append(info)
    pool = threadpool.ThreadPool(30)
    requests = threadpool.makeRequests(gen_data, data_lst, write_list)
    ret_lines = [pool.putRequest(req) for req in requests]
    pool.wait()

