import tqdm
import yaml
import argparse
import cv2
import numpy as np
from easydict import EasyDict
import torch

from tools.load import LoadMADSData
from tools.common import project_3d_to_2d
from tools.utils import check_occlusion
from tools.augmentation import Cutout, HideNSeek


def get_2d_pose(pose_3d, calibs):
    K = np.array(calibs['intrinsics'])
    T = np.array(calibs['translation'])
    R = np.array(calibs['rotation'])

    pose_2d = project_3d_to_2d(pose_3d, K, R, T)
    return pose_2d


def draw_pose(img, pose_2d, mask=None):

    pose_2d = pose_2d.astype(np.int32)
    if mask is not None:
        pose_2d = pose_2d * mask
    for i in range(pose_2d.shape[0]):
        cv2.circle(img, tuple(pose_2d[i][:2]), 2, (0, 0, 255), -1)

    return img


def visualize_pose(left_img, right_img, meta):
    # Load the 3D pose
    pose_3d = np.array(meta['pose_3d'])

    # Draw the pose on the left image
    hns = HideNSeek(4)
    #hns = Cutout(6, 40)
    left_img, mask_l = hns(left_img)
    right_img, mask_r = hns(right_img)

    pose_2d_left = get_2d_pose(pose_3d, meta['cam_left'])
    pose_2d_right = get_2d_pose(pose_3d, meta['cam_right'])

    mask_l = check_occlusion(pose_2d_left, mask_l)
    mask_r = check_occlusion(pose_2d_right, mask_r)
    mask = np.logical_and(mask_l, mask_r)

    left_img = draw_pose(left_img, pose_2d_left, mask)
    right_img = draw_pose(right_img, pose_2d_right, mask)
    img_stereo = np.concatenate((left_img, right_img), axis=1)

    # Save the results
    cv2.imwrite('example_image.png', img_stereo)
    exit()
    # cv2.imshow("img", img_stereo)
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     print("quit display")
    #     exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/mads_2d.yaml",
                        help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    MADS = LoadMADSData("data/MADS_extract/valid",
                        config.MODEL.IMAGE_SIZE)

    for idx, (left_img, right_img, meta) in tqdm.tqdm(enumerate(MADS),
                                                      total=len(MADS)):
        visualize_pose(np.array(left_img), np.array(right_img), meta)
