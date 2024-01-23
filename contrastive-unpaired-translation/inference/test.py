"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from tqdm import tqdm
from math import ceil
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
import torchvision.transforms as transforms
from scipy.ndimage import zoom

from test_options import TestOptions
from models import create_model
# from models import create_model

# def zscore(vol):
#     vol_min = np.percentile(vol, 5)
#     vol_max = np.percentile(vol, 95)
#     vol = np.clip(vol, vol_min, vol_max)
#     img_mean = vol.mean()
#     img_std = vol.std()
#     vol = (vol - img_mean) / (img_std + 1e-12)
#     return vol

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

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 2    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.gpu_ids = [0]
    opt.name = "cut0.001_16_256_AtoB_resnet_9blocks_basic"
    
    save_path = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating result directory', save_path)
    os.makedirs(save_path, exist_ok=True)
    label_path = '/home/tx-deepocean/Data/Project/crossMoDA/data/crossmoda23_training/TrainingSource/label'
    predata_path = '/home/tx-deepocean/Data/Project/crossMoDA/IARSeg/train/tools/checkpoints/predata_fakeT2_epoch90/source'
    os.makedirs(predata_path, exist_ok=True)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model = create_model(opt)     # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # model.eval()

    batch_size = opt.batch_size
    crop_size = np.array([256, 256])
    for pid_id, pid in enumerate(sorted(os.listdir(opt.dataroot))[:7]):
        print(f'processing image: {pid}')
        sitk_img = sitk.ReadImage(os.path.join(opt.dataroot, pid))
        img_arr = sitk.GetArrayFromImage(sitk_img)
        spacing_vol = sitk_img.GetSpacing()[::-1]
        depth, w, h = img_arr.shape
        start_w = int(3 * w / 16); end_w = int(13 * w / 16)
        start_h = int(3 * h / 16); end_h = int(13 * h / 16)

        sitk_label = sitk.ReadImage(os.path.join(label_path, pid.replace('ceT1', 'Label')))
        label_arr = sitk.GetArrayFromImage(sitk_label)

        cropped_img = img_arr[:, start_w:end_w, start_h:end_h]
        cropped_label = label_arr[:, start_w:end_w, start_h:end_h]
        assert (np.array(cropped_img.shape) == np.array(cropped_label.shape)).all()

        cropped_real_A = np.zeros((depth, crop_size[0], crop_size[1]))
        cropped_fake_B = np.zeros((depth, crop_size[0], crop_size[1]))
        cropped_label = zoom(cropped_label, np.array(np.array((depth, crop_size[0], crop_size[1])) / np.array(cropped_label.shape)), order=0)
        
        # select valid scans
        pid_scans_idxes = []
        for idx in range(depth):
            scan = cropped_img[idx]
            if (scan.max() - scan.min()) < 10:
                continue
            pid_scans_idxes.append(idx)
        pid_scans_idxes = np.array(pid_scans_idxes)
        
        # inference
        for i in range(ceil(len(pid_scans_idxes) / batch_size)):
            batch_A = []
            batch_idxes = pid_scans_idxes[(i * batch_size):((i+1) * batch_size)]
            for idx in batch_idxes:
                A_img = cropped_img[idx]
                # A_img = zscore(A_img)
                A_img = normalize(A_img)
                A_img = zoom(A_img, np.array(np.array(crop_size) / np.array(A_img.shape)), order=1)

                cropped_real_A[idx] = A_img

                A_img = torch.from_numpy(A_img).float()[None] # channel first
                batch_A.append(A_img[None])
                
            batch_A = torch.cat(batch_A, dim=0)
            assert len(batch_A.shape) == 4 # b, c, w, h
            with torch.no_grad():
                if pid_id == 0 and i == 0:
                    # model.data_dependent_initialize(batch_A)
                    model.setup(opt)  # regular setup: load and print networks; create schedulers
                    model.eval()
                model.set_input(batch_A)
                model.test()
                visuals = model.get_current_visuals()
                batch_fake_B = visuals['fake_B'].clamp(-1.0, 1.0).cpu().numpy()
            assert len(batch_fake_B) == len(batch_idxes)
            for k, idx in enumerate(batch_idxes):
                fake_B = batch_fake_B[k, 0]
                cropped_fake_B[idx] = fake_B
                # print('00: ', len(np.unique(fake_B)))
                # print('11', len(np.unique(img_fake_B)))
        
        assert (np.array(cropped_real_A.shape) == np.array(cropped_fake_B.shape)).all()
        assert (np.array(cropped_real_A.shape) == np.array(cropped_label.shape)).all()
        assert (np.array(cropped_real_A.shape) == np.array([depth, crop_size[0], crop_size[1]])).all()
        x_flip, y_flip, z_flip = is_flip(sitk_img.GetDirection())
        
        if x_flip:
            print(pid, ', x_flip:', x_flip)
            cropped_real_A = np.ascontiguousarray(np.flip(cropped_real_A, 2))
            cropped_fake_B = np.ascontiguousarray(np.flip(cropped_fake_B, 2))
            cropped_label = np.ascontiguousarray(np.flip(cropped_label, 2))
        if y_flip:
            print(pid, ', y_flip:', y_flip)
            cropped_real_A = np.ascontiguousarray(np.flip(cropped_real_A, 1))
            cropped_fake_B = np.ascontiguousarray(np.flip(cropped_fake_B, 1))
            cropped_label = np.ascontiguousarray(np.flip(cropped_label, 1))
        if z_flip:
            print(pid, ', z_flip:', z_flip)
            cropped_real_A = np.ascontiguousarray(np.flip(cropped_real_A, 0))
            cropped_fake_B = np.ascontiguousarray(np.flip(cropped_fake_B, 0))
            cropped_label = np.ascontiguousarray(np.flip(cropped_label, 0)) 
        # np.savez_compressed(
        #     os.path.join(predata_path, pid.replace('.nii.gz', '.npz')),
        #     realT1=cropped_real_A,
        #     fakeT2=cropped_fake_B,
        #     seg=cropped_label,
        #     src_spacing=np.array(spacing_vol),
        # ) 

        cropped_fake_B = 255 * (cropped_fake_B  + 1) / 2
        img_fake_B_itk = sitk.GetImageFromArray((1 * cropped_fake_B).astype(np.int32))
        # img_fake_B_itk.CopyInformation(sitk_img)
        sitk.WriteImage(img_fake_B_itk, os.path.join(save_path, pid.replace('.nii.gz', '')+'_fake_B.nii.gz'))
        cropped_real_A = 255 * (cropped_real_A + 1) / 2
        img_real_A_itk = sitk.GetImageFromArray((1 * cropped_real_A).astype(np.int32))
        # img_real_A_itk.CopyInformation(sitk_img)
        sitk.WriteImage(img_real_A_itk, os.path.join(save_path, pid.replace('.nii.gz', '')+'_real_A.nii.gz'))

