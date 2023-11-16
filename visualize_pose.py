import tqdm
import os
import cv2
import numpy as np

from tools.load import LoadMDASData
from tools.common import world_to_camera, camera_to_image


def undistort_image(image, K, dist_coeffs, new_K=np.array([])):
    if new_K.size == 0:
        new_K = K
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_K)

    return undistorted_image


def project_3d_to_2d(pose_3d, K, R, T):
    # Transform the 3D points into the camera coordinate system
    pose_2d = world_to_camera(pose_3d, R, T)
    pose_2d = camera_to_image(pose_2d, K)

    return pose_2d


def draw_pose(img, pose_2d, calibs):
    K = np.array(calibs['intrinsics'])
    T = np.array(calibs['translation'])
    R = np.array(calibs['rotation'])
    dist_coeffs = np.array(calibs['distortion_coeffs'])

    pose_2d = project_3d_to_2d(pose_2d, K, R, T)
    img = undistort_image(img, K, dist_coeffs)

    pose_2d = pose_2d.astype(np.int32)
    for i in range(pose_2d.shape[0]):
        cv2.circle(img, tuple(pose_2d[i][:2]), 2, (0, 0, 255), -1)

    return img


def visualize_pose(idx, metadata, output_path):
    # Create a directory to save results
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the images
    img_left = cv2.imread(metadata['left_img_path'])
    img_right = cv2.imread(metadata['right_img_path'])

    # Load the calibration information
    calibs_left = metadata['cam_left']
    calibs_right = metadata['cam_right']

    # Load the 2D pose
    pose_3d = np.array(metadata['pose_3d'])

    # Draw the pose on the left image
    img_left = draw_pose(img_left, pose_3d, calibs_left)
    img_right = draw_pose(img_right, pose_3d, calibs_right)

    # Save the results
    img_stereo = np.concatenate((img_left, img_right), axis=1)
    cv2.imwrite(os.path.join(output_path, f"stereo_{idx:04d}.jpg"), img_stereo)


if __name__ == "__main__":
    MADS = LoadMDASData("data/MADS_training")

    for idx, metadata in tqdm.tqdm(enumerate(MADS), total=len(MADS)):
        visualize_pose(idx, metadata, "output")
