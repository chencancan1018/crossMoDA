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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_path', type=str, default='/home/tx-deepocean/Data/Project/crossMoDA/data/crossmoda23_training')
    parser.add_argument('--out_path', type=str, default='./checkpoints/predata_fakeT2_round1')

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

def gen_source_data(info):
    
    try:
        pid, vol_file, out_dir, seg_file, sitk_lock = info
        save_file = os.path.join(out_dir, f'{pid}.npz')
        sitk_lock.acquire()
        vol, sitk_img, spacing_vol = load_nii(vol_file)
        sitk_lock.release()

        if  not os.path.exists(seg_file):
            return None
        seg, _, _ = load_nii(seg_file)
        seg = seg.astype(np.uint8)    

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

        vol_shape = np.array(vol.shape)
        seg_shape = np.array(seg.shape)
        if np.any(vol_shape != seg_shape):
            print('pid vol shape != seg shape: ', pid)
            return None

        np.savez_compressed(
            save_file,
            vol=vol,
            seg=seg,
            src_spacing=np.array(spacing_vol),
        )
        print(f'{pid} successed')
        return save_file
    except:
        traceback.print_exc()
        sitk_lock.release()
        print(f'{pid} failed')
        return None
    
def gen_target_data(info):
    
    try:
        pid, vol_file, out_dir, sitk_lock = info
        save_file = os.path.join(out_dir, f'{pid}.npz')
        sitk_lock.acquire()
        vol, sitk_img, spacing_vol = load_nii(vol_file)
        sitk_lock.release()  

        x_flip, y_flip, z_flip = is_flip(sitk_img.GetDirection())
        print(pid, sitk_img.GetSize(), sitk_img.GetDirection(), x_flip, y_flip, z_flip)
        if x_flip:
            vol = np.ascontiguousarray(np.flip(vol, 2))
        if y_flip:
            vol = np.ascontiguousarray(np.flip(vol, 1))
        if z_flip:
            vol = np.ascontiguousarray(np.flip(vol, 0)) 

        np.savez_compressed(
            save_file,
            vol=vol,
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
    source_dir = os.path.join(args.src_path, 'TrainingSource')
    target_dir = os.path.join(args.tgt_path, 'TrainingTarget')
    source_out_dir = os.path.join(args.out_path, 'source')
    target_out_dir = os.path.join(args.out_path, 'target')

    os.makedirs(source_out_dir, exist_ok=True)
    data_lst = []
    for pid in sorted(os.listdir(os.path.join(source_dir, 'img'))):
        vol_file = os.path.join(source_dir, 'img', pid)
        seg_file = os.path.join(source_dir, 'label', pid.replace('ceT1', 'Label'))
        info = [pid.replace('.nii.gz', ''), vol_file, source_out_dir, seg_file, sitk_lock]
        data_lst.append(info)
    pool = threadpool.ThreadPool(30)
    requests = threadpool.makeRequests(gen_source_data, data_lst, write_list)
    ret_lines = [pool.putRequest(req) for req in requests]
    pool.wait()

    print(f'finshed {len(data_lst)} patient.')
    fold_num = 5
    gen_lst(source_out_dir, fold_num=fold_num)

    os.makedirs(target_out_dir, exist_ok=True)
    data_lst = []
    for pid in sorted(os.listdir(target_dir)):
        vol_file = os.path.join(target_dir, pid)
        info = [pid.replace('.nii.gz', ''), vol_file, target_out_dir, sitk_lock]
        data_lst.append(info)
    pool = threadpool.ThreadPool(30)
    requests = threadpool.makeRequests(gen_target_data, data_lst, write_list)
    ret_lines = [pool.putRequest(req) for req in requests]
    pool.wait()

    print(f'finshed {len(data_lst)} patient.')
    fold_num = 5
    gen_lst(target_out_dir, fold_num=fold_num)
    
