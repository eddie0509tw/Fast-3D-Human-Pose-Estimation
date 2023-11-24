import cv2
import numpy as np


def world_to_camera(points, R, T):
    # Construct the transformation matrix
    Rt = np.concatenate((R, T), axis=1)
    Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)

    # Transform the 3D points into the camera coordinate system
    points_hom = np.vstack((points.T,
                            np.ones((1, points.shape[0]))))

    points_hom = Rt @ points_hom

    return points_hom[:3].T


def camera_to_image(points, K):
    points_2d = K @ points.T

    points_2d = points_2d.T
    points_2d[:, :2] /= points_2d[:, 2:]

    return points_2d


def project_3d_to_2d(pose_3d, K, R, T):
    # Transform the 3D points into the camera coordinate system
    pose_2d = world_to_camera(pose_3d, R, T)
    pose_2d = camera_to_image(pose_2d, K)

    return pose_2d


def undistort_image(image, K, dist_coeffs, new_K=np.array([])):
    if new_K.size == 0:
        new_K = K.copy()
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_K)

    return undistorted_image
