from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tools.load import LoadMDASData


class MADS(Dataset):  # Fix: Change dataset to Dataset

    def __init__(self, img_dir_poses):
        self.img_dir_poses = img_dir_poses
        MADS_load = LoadMDASData(self.img_dir_poses)
        self.meta_data = MADS_load.metadata

    def __getitem__(self, index):
        self.img_left = self.meta_data[index]['left_img_path']
        self.img_right = self.meta_data[index]['right_img_path']
        self.cam_left = self.meta_data[index]['cam_left']
        self.cam_right = self.meta_data[index]['cam_right']
        self.img_poses = self.meta_data[index]['pose_3d']

        image_left = Image.open(self.img_left)
        image_left = np.asarray(image_left).transpose(2, 0, 1).copy()
        image_left = torch.from_numpy(image_left).float()

        image_right = Image.open(self.img_right)
        image_right = np.asarray(image_right).transpose(2, 0, 1).copy()
        image_right = torch.from_numpy(image_right).float()

        return image_left, image_right, self.cam_left, \
            self.cam_right, self.img_poses

    def __len__(self):
        return len(self.meta_data)

if __name__ == '__main__':
    path = "/home/eddieshen/ESE546/3d_HPE/data/MADS_training/"
    MADS_dataset = MADS(path)
    image_left, image_right, cam_left, cam_right, img_poses = MADS_dataset[0]
    print(image_left.shape)
    print(image_right.shape)
    print(cam_left)