import os
import shutil
import glob
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk

def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    sitk_img = reader.Execute()
    data_np = sitk.GetArrayFromImage(sitk_img)
    spacing = sitk_img.GetSpacing()
    spacing = spacing[::-1]
    return data_np, sitk_img, spacing

def load_nii(nii_path):
    sitk_img = sitk.ReadImage(nii_path)
    spacing = sitk_img.GetSpacing()
    spacing = spacing[::-1]
    origin_coord = sitk_img.GetOrigin()
    data_np = sitk.GetArrayFromImage(sitk_img)
    return data_np, sitk_img, spacing

def find_valid_region(mask):
    binary_mask = mask.copy()
    binary_mask = (binary_mask > 0)
    nonzero_points = np.argwhere(binary_mask)
    pmin = np.min(nonzero_points, axis=0)
    pmax = np.max(nonzero_points, axis=0)
    return pmin, pmax

def find_next_nonzero_slice(idx_z, mask, up_margin):
    binary_mask = mask.copy()
    next_idx_z = idx_z + 1
    while next_idx_z <= up_margin:
        if np.sum(binary_mask[next_idx_z]) > 0:
            break
        else:
            next_idx_z += 1
    return next_idx_z

def single_data_interpolate(dcm_path, mask_path):
    vol, sitk_vol, spacing = load_scans(dcm_path)
    mask, sitk_mask, _ = load_nii(mask_path)
    mask = mask.astype(np.uint8)

    assert np.any(vol.shape == mask.shape), "vol and mask shape dismatch!!" 
    shape_z, shape_y, shape_x = vol.shape

    # 利用标注文件的上下层ROI，循环插值中间缺失ROI
    pmin, pmax = find_valid_region(mask)
    range_z = [pmin[0], pmax[0]]
    idx_z = pmin[0]
    while idx_z <= pmax[0]:
        mask_slice = mask[idx_z]
        if np.sum(mask_slice) > 0:
            idx_z += 1
        else:
            next_nonzero_idx_z = find_next_nonzero_slice(idx_z, mask, pmax[0])
            if next_nonzero_idx_z >= (idx_z + 1):
                if (next_nonzero_idx_z - idx_z) <= 5: 
                    # 若缺失层数较少，使用上下层ROI循环插值中间缺失ROI
                    cur_nonzero_idx_z = idx_z - 1
                    cur_mask = np.zeros((2, shape_y, shape_x), dtype=np.uint8)
                    cur_mask[0, :, :] = mask[cur_nonzero_idx_z].copy()
                    cur_mask[1, :, :] = mask[next_nonzero_idx_z].copy()
                    target_shape = [next_nonzero_idx_z - cur_nonzero_idx_z + 1, shape_y, shape_x]
                    zoomed = zoom(cur_mask, np.array(np.array(target_shape) / cur_mask.shape), order=1)
                    mask[idx_z:next_nonzero_idx_z, :, :] = zoomed[1:-1, :, :]
                    mask[mask > 0.25] = 1
                else:
                    # 若缺失层数较多，使用之前所有ROI插值当前缺失ROI
                    cur_nonzero_idx_z = pmin[0]
                    cur_mask = np.zeros(((idx_z-pmin[0]+1), shape_y, shape_x))
                    cur_mask[:-1, :, :] = mask[pmin[0]:idx_z, :, :].copy()
                    cur_mask[-1, :, :] = mask[next_nonzero_idx_z, :, :].copy()
                    target_shape = [next_nonzero_idx_z - pmin[0] + 1, shape_y, shape_x]
                    zoomed = zoom(cur_mask, np.array(np.array(target_shape) / cur_mask.shape), order=1)
                    mask[idx_z:next_nonzero_idx_z, :, :] = zoomed[(idx_z-pmin[0]):-1, :, :]
                    mask[mask > 0.01] = 1
                idx_z = next_nonzero_idx_z 
            else:
                idx_z += 1
            
    return sitk_vol, mask.astype(np.uint8)


if __name__ == "__main__":
    datadir = './20220905/'
    maskdir = './annotation'
    outdir = "./TrainMask"
    pids = sorted(os.listdir(datadir))

    # 整理数据，标注文件重命名为文件名
    os.makedirs(maskdir, exist_ok=True)
    for pid in pids:
        orig_mask_path = glob.glob(os.path.join(datadir, pid, '*.nii'))[0]
        mask_name = os.path.basename(orig_mask_path)
        print(pid, mask_name)
        desti_mask_path = os.path.join(maskdir, pid+'_mask.nii')
        shutil.copy(orig_mask_path, maskdir)
        os.rename(os.path.join(maskdir, mask_name), desti_mask_path)

    # 通过插值补充缺失数据
    # 若缺失层数较多，插值效果较差
    os.makedirs(outdir, exist_ok=True)
    for pid in pids:
        print("start to process {}.".format(pid))
        dcm_path = os.path.join(datadir, pid)
        mask_path = os.path.join(maskdir, pid+'_mask.nii')
        sitk_vol, mask = single_data_interpolate(dcm_path, mask_path)
        sitk_save = sitk.GetImageFromArray(mask)
        sitk_save.CopyInformation(sitk_vol)
        sitk.WriteImage(sitk_save, os.path.join(outdir, pid+'_mask.nii.gz'))
    print("Finished!")

