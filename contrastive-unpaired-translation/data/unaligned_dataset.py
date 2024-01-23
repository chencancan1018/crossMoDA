import os.path
from data.base_dataset import BaseDataset, MonaiTransform2D
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import util.util as util
import numpy as np
from scipy.ndimage import zoom


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        # self.dir_A = os.path.join(opt.dataroot, 'ADir_zscore')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, 'BDir_zscore')  # create a path '/path/to/data/trainB'
        self.dir_A = os.path.join(opt.dataroot, 'ADir_01')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, 'BDir_01')  # create a path '/path/to/data/trainB'

        # if opt.phase == "test" and not os.path.exists(self.dir_A) \
        #    and os.path.exists(os.path.join(opt.dataroot, "valA")):
        #     self.dir_A = os.path.join(opt.dataroot, "valA")
        #     self.dir_B = os.path.join(opt.dataroot, "valB")

        print(f"===========start to read {self.dir_A}===========")
        self.A_paths = sorted(self.load_npz_scans(self.dir_A))
        print(f"===========start to read {self.dir_B}===========")
        self.B_paths = sorted(self.load_npz_scans(self.dir_B))
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        print(f"=========== Complete data initialize, A_size:{self.A_size} and {self.B_size} ===========")
        btoA = opt.direction == 'BtoA'

        self.show_idx = 0
        self.crop_size = np.array((opt.crop_size, opt.crop_size))
        self.transform_A = MonaiTransform2D(opt)
        self.transform_B = MonaiTransform2D(opt)

        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B

    def load_npz_scans(self, dir):
        paths = sorted(os.listdir(dir))
        # paths = [p for p in paths if 'etz' in p]
        paths = [os.path.join(dir, p) for p in  paths]
        return paths

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        data_A = np.load(A_path, allow_pickle=True)
        A_img = data_A["img"]
        data_B = np.load(B_path, allow_pickle=True)
        B_img = data_B["img"]

        # resize
        if np.any(np.array(A_img.shape) != self.crop_size):
            A_img = zoom(A_img, np.array(self.crop_size / np.array(A_img.shape)), order=1)
        if np.any(np.array(B_img.shape) != self.crop_size):
            B_img = zoom(B_img, np.array(self.crop_size / np.array(B_img.shape)), order=1)

        A_img = (A_img - 0.5) / 0.5 # [0, 1] to [-1, 1]
        B_img = (B_img - 0.5) / 0.5 # [0, 1] to [-1, 1]
        
        # numpy to tensor
        A_img = torch.from_numpy(A_img).float()[None]
        B_img = torch.from_numpy(B_img).float()[None]

        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        # is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        # modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
