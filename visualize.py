import tqdm
import yaml
import argparse
import cv2
import numpy as np
from easydict import EasyDict

from tools.load import LoadMADSData
from tools.common import project_3d_to_2d


def draw_pose(img, pose_3d, calibs):
    K = np.array(calibs['intrinsics'])
    T = np.array(calibs['translation'])
    R = np.array(calibs['rotation'])

    pose_2d = project_3d_to_2d(pose_3d, K, R, T)

    pose_2d = pose_2d.astype(np.int32)
    for i in range(pose_2d.shape[0]):
        cv2.circle(img, tuple(pose_2d[i][:2]), 2, (0, 0, 255), -1)

    return img


def visualize_pose(left_img, right_img, meta):
    # Load the 3D pose
    pose_3d = np.array(meta['pose_3d'])

    # Draw the pose on the left image
    img_left = draw_pose(left_img, pose_3d, meta['cam_left'])
    img_right = draw_pose(right_img, pose_3d, meta['cam_right'])

    # Save the results
    img_stereo = np.concatenate((img_left, img_right), axis=1)
    cv2.imshow("img", img_stereo)
    key = cv2.waitKey(0)
    if key == ord('q'):
        print("quit display")
        exit(1)


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
        visualize_pose(left_img, right_img, meta)
