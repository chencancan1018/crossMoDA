import os
import numpy as np 
from skimage.measure import label, regionprops
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
import threadpool
import threading
from queue import Queue

def get_connected_regions(mask, intensity=None):
    labeled = label(mask)
    regions = regionprops(labeled, intensity_image=intensity)
    return labeled, regions

def binarization_thres(image_arr, thres):
    arr = image_arr.copy()
    arr[arr >= thres] = 1
    arr[arr < thres] = 0 
    return arr

def random3D(bbox):
    x1 = np.random.choice(range(bbox[0], bbox[3]))
    x2 = np.random.choice(range(bbox[1], bbox[4]))
    x3 = np.random.choice(range(bbox[2], bbox[5]))
    return x1, x2, x3

def coord_round(x, radius, upper):
    assert len(x) == len(radius)
    assert len(x) == len(upper)
    cent = [0] * len(x)
    for i in range(len(x)):
        if x[i] < radius[i]:
            cent[i] = np.max([x[i], radius[i]]) 
        elif x[i] > upper[i]:
            cent[i] = np.min([x[i], upper[i]])
        else:
            cent[i] = x[i]
    return cent

def resample(img_arr, new_shape, order=3):
    resize_factor = new_shape / img_arr.shape
    img_arr = zoom(img_arr, resize_factor, mode='nearest', order=order)
    return img_arr

def load_nii(nii_path):
    tmp_img = sitk.ReadImage(nii_path)
    spacing = tmp_img.GetSpacing()
    spacing = spacing[::-1]
    data_np = sitk.GetArrayFromImage(tmp_img)
    return data_np, tmp_img, spacing

def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    vol = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing = spacing[::-1]
    return vol, img, spacing

def dcm_to_nii(info):
    
    pid, dcm_path, save_path = info
    print('start to converse {}'.format(pid))
    _, img, _ = load_scans(dcm_path)
    sitk.WriteImage(img, save_path)


