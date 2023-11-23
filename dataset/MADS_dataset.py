from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from tools.load import LoadMDASData
from tools.visualize_pose import project_3d_to_2d, cropping
import cv2


class MADS(Dataset):  # Fix: Change dataset to Dataset

    def __init__(self, img_dir_poses, target_size=(256, 256)):
        self.img_dir_poses = img_dir_poses
        MADS_load = LoadMDASData(self.img_dir_poses)
        self.meta_data = MADS_load.metadata
        self.target_size = target_size

    def __getitem__(self, index):
        self.img_left = self.meta_data[index]['left_img_path']
        self.img_right = self.meta_data[index]['right_img_path']
        self.cam_left = self.meta_data[index]['cam_left']
        self.cam_right = self.meta_data[index]['cam_right']
        self.pose_3d = self.meta_data[index]['pose_3d']
        self.pose_3d = np.array(self.pose_3d)

        image_left = Image.open(self.img_left)
        image_left = np.asarray(image_left)
        image_left, gt_2d_l, _, left_K = cropping(
                                            image_left, self.pose_3d,
                                            self.cam_left,
                                            target_size=self.target_size)
        image_left = torch.from_numpy(image_left.transpose(2, 0, 1)).float()

        image_right = Image.open(self.img_right)
        image_right = np.asarray(image_right)
        image_right, gt_2d_r, _, right_K = cropping(
                                    image_right, self.pose_3d,
                                    self.cam_right,
                                    target_size=self.target_size)
        image_right = torch.from_numpy(image_right.transpose(2, 0, 1)).float()

        gt_3d, gt_2d_l, gt_2d_r = self.process_gt_poses(
                                            self.pose_3d, self.cam_left,
                                            self.cam_right, left_K, right_K)
        left_proj, right_proj = self.get_projection_matrix(
            self.cam_left, self.cam_right, left_K, right_K)

        return image_left, image_right, left_proj, \
            right_proj, gt_3d, gt_2d_l, gt_2d_r

    def __len__(self):
        return len(self.meta_data)

    def get_projection_matrix(self, cam_left, cam_right, left_K, right_K):
        left_K = torch.from_numpy(left_K).to(torch.float32)
        left_rot = torch.tensor(cam_left['rotation'])
        left_trans = torch.tensor(cam_left['translation'])
        right_K = torch.from_numpy(right_K).to(torch.float32)
        right_rot = torch.tensor(cam_right['rotation'])
        right_trans = torch.tensor(cam_right['translation'])

        left_proj = left_K @ torch.cat((left_rot, left_trans), 1)
        right_proj = right_K @ torch.cat((right_rot, right_trans), 1)

        return left_proj, right_proj

    def process_gt_poses(self, pose_3d, cam_left, cam_right, left_K, right_K):
        left_rot = np.array(cam_left['rotation'])
        left_trans = np.array(cam_left['translation'])
        right_rot = np.array(cam_right['rotation'])
        right_trans = np.array(cam_right['translation'])

        pose_2d_l = project_3d_to_2d(pose_3d, left_K, left_rot, left_trans)
        pose_2d_r = project_3d_to_2d(pose_3d, right_K, right_rot, right_trans)

        return torch.from_numpy(pose_3d).float(), \
            torch.from_numpy(pose_2d_l).float()[..., :2], \
            torch.from_numpy(pose_2d_r).float()[..., :2]


# test for MADS dataset
if __name__ == '__main__':
    path = "/home/eddieshen/ESE546/3d_HPE/data/MADS_training/"
    MADS_dataset = MADS(path)
    image_left, image_right, left_proj, right_proj, \
        gt_3d, gt_2d_l, gt_2d_r = MADS_dataset[0]
   
    print(image_left.size())
    print(type((cv2.resize(image_left.permute(1, 2, 0).numpy(), (256, 256)))))
    print(gt_3d.size())
    print(gt_2d_l)
    print(gt_2d_r)
