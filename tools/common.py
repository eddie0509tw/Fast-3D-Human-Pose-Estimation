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
