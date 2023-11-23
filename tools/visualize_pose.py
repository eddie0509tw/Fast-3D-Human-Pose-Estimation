import tqdm
import os
import cv2
import numpy as np

from tools.load import LoadMDASData
from tools.common import world_to_camera, camera_to_image, adjust_intrinsic


def undistort_image(image, K, dist_coeffs, new_K=np.array([])):
    if new_K.size == 0:
        new_K = K
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_K)

    return undistorted_image


def project_3d_to_2d(pose_3d, K, R, T):
    # Transform the 3D points into the camera coordinate system
    pose_2d = world_to_camera(pose_3d, R, T)
    pose_2d = camera_to_image(pose_2d, K)
    print(pose_2d)

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

        cv2.putText(
            img, str(i+1), tuple(map(int, pose_2d[i][:2])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img


def draw_pose_cropped(img, pose_3d, calibs):
    K = np.array(calibs['intrinsics'])
    dist_coeffs = np.array(calibs['distortion_coeffs'])

    img = undistort_image(img, K, dist_coeffs)

    img, pose_2d = cropping(img, pose_3d, calibs)

    pose_2d = pose_2d.astype(np.int32)
    for i in range(pose_2d.shape[0]):
        cv2.circle(img, tuple(pose_2d[i][:2]), 2, (0, 0, 255), -1)

        cv2.putText(
            img, str(i+1), tuple(map(int, pose_2d[i][:2])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return img


def visualize_pose(idx, metadata, output_path, is_crop=False):
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
    if is_crop:
        img_left = draw_pose_cropped(img_left, pose_3d, calibs_left)
        img_right = draw_pose_cropped(img_right, pose_3d, calibs_right)
    else:
        img_left = draw_pose(img_left, pose_3d, calibs_left)
        img_right = draw_pose(img_right, pose_3d, calibs_right)

    # Save the results
    img_stereo = np.concatenate((img_left, img_right), axis=1)
    if is_crop:
        cv2.imwrite(os.path.join(
            output_path, f"cropped_stereo_{idx:04d}.jpg"), img_stereo)
    else:
        cv2.imwrite(os.path.join(
            output_path, f"stereo_{idx:04d}.jpg"), img_stereo)


def cropping(img, pose_3d, calibs, target_size=(256, 256)):
    h, w = np.array(img).shape[:2]

    tw, th = target_size
    # Calculate the side length of the square crop
    side_length = min(h, w)

    # Calculate the cropping margins for both sides
    margin_x = (w - side_length) // 2
    margin_y = (h - side_length) // 2
    # Crop the image to the square
    cropped_img = img[
                    margin_y:h - margin_y,
                    margin_x:w - margin_x
                    ]

    cropped_img = cv2.resize(cropped_img, target_size)

    shift_topleft = - np.array([margin_x, margin_y])
    K = np.array(calibs['intrinsics'])
    K_new = adjust_intrinsic(K, side_length, target_size, shift_topleft)

    pose_2d = project_3d_to_2d(
        pose_3d, K_new,
        np.array(calibs['rotation']),
        np.array(calibs['translation']))

    pose_2d = pose_2d[..., :2]

    pose_2d = pose_2d[
        (pose_2d[:, 0] >= 0) & (pose_2d[:, 0] < tw) &
        (pose_2d[:, 1] >= 0) & (pose_2d[:, 1] < th)]

    return cropped_img, pose_2d


if __name__ == "__main__":
    MADS = LoadMDASData("../data/test")

    for idx, metadata in tqdm.tqdm(enumerate(MADS), total=len(MADS)):
        visualize_pose(idx, metadata, "output_cropped", is_crop=True)
