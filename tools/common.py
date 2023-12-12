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


def get_projection_matrix(K, R, T):
    P = K @ np.hstack((R, T))
    P = np.vstack((P, np.array([0, 0, 0, 1])))

    return P


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


def triangulation(P1, P2, pts1, pts2):
    pts3D = []

    for pt1, pt2 in zip(pts1, pts2):
        pt1_ = np.array([pt1[0], pt1[1], 1])
        pt2_ = np.array([pt2[0], pt2[1], 1])

        pt1_ = np.cross(pt1_, np.identity(pt1_.shape[0]) * -1)
        pt2_ = np.cross(pt2_, np.identity(pt2_.shape[0]) * -1)

        M1 = np.array([pt1[1] * P1[2] - P1[1], P1[0] - pt1[0] * P1[2]])
        M2 = np.array([pt2[1] * P2[2] - P2[1], P2[0] - pt2[0] * P2[2]])
        M = np.vstack((M1, M2))

        e, v = np.linalg.eig(M.T @ M)
        idx = np.argmin(e)
        pt3 = v[:, idx]
        pt3 = (pt3 / pt3[-1])[:3]
        pts3D.append(pt3)

    return np.array(pts3D)
