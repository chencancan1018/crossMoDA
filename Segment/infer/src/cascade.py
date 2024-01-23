import os
import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy.ndimage import binary_dilation, binary_erosion

def compute_dist(mask1, mask2, label=1):
    temp1 = (mask1 == label) * 1
    temp2 = (mask2 == label) * 1
    gap = temp1 - temp2
    dist = (gap != 0).sum() / (temp1.sum() + temp2.sum() + 1)
    return dist

if __name__ == "__main__":
    dir_path = './crop_round9_0704_epoch190/'
    folds = sorted(os.listdir(dir_path))
    # folds = [f for f in folds if f.startswith('Training')]
    # out_path = dir_path + 'CascadeTrainingTarget'
    folds = [f for f in folds if f.startswith('validation')]
    out_path = dir_path + 'CascadeValidation'
    os.makedirs(out_path, exist_ok=True)
    assert len(folds) == 8

    pids = sorted(os.listdir(os.path.join(dir_path, folds[0])))
    # bad_case = ['crossmoda2023_ukm_168.nii.gz']
    bad_case = []
    for pid in pids:
        sitk_mask0 = sitk.ReadImage(os.path.join(dir_path, folds[0], pid))
        mask0 = sitk.GetArrayFromImage(sitk_mask0)
        mask1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir_path, folds[1], pid)))
        mask2 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir_path, folds[2], pid)))
        mask3 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir_path, folds[3], pid)))
        mask4 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir_path, folds[4], pid)))
        mask5 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir_path, folds[5], pid)))
        mask6 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir_path, folds[6], pid)))
        mask7 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(dir_path, folds[7], pid)))

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
                print(pid, f"cochlea seg is empty!!")
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
                        raise KeyError(f'{pid} with wrong seg result!')
                else:
                    temp = (labeled> 0) * 1
            else:
                print(pid, 'lesion num: 0')
                
            temp[out == 3] = 1
            out = out * temp
            out = out.astype(np.uint8)

        # save
        out_itk = sitk.GetImageFromArray(out)
        out_itk.CopyInformation(sitk_mask0)
        sitk.WriteImage(out_itk, os.path.join(out_path, pid))
    print(f"=================bad case: {bad_case}===================")

        

    


