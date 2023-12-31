import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np


class Rectification:
    def __init__(self, calib_file_path):
        self.calibs = dict()
        self._parse_calibration_data(calib_file_path)

    def _parse_calibration_data(self, calibration_file_path):
        # Load calibration data
        with open(calibration_file_path, 'r') as f:
            calib_data = yaml.safe_load(f)

        # Extract camera intrinsic matrix and distortion coefficients
        left_camera_intrinsic = np.array(calib_data['cam0']['intrinsics'])
        left_dist_coeffs = np.array(calib_data['cam0']['distortion_coeffs'])
        right_camera_intrinsic = np.array(calib_data['cam1']['intrinsics'])
        right_dist_coeffs = np.array(calib_data['cam1']['distortion_coeffs'])
        trans_matrix = np.array(calib_data['cam1']['T_cn_cnm1']).reshape(4, 4)

        # Create camera intrinsic matrix with four parameters
        self.calibs = {
            'left': {
                'K': self._gen_intrinsic(left_camera_intrinsic),
                'dist_coeffs': left_dist_coeffs
            },
            'right': {
                'K': self._gen_intrinsic(right_camera_intrinsic),
                'dist_coeffs': right_dist_coeffs
            },
            'T': trans_matrix
        }

    @staticmethod
    def _gen_intrinsic(camera_intrinsic):
        K = np.array([
            [camera_intrinsic[0], 0, camera_intrinsic[2]],
            [0, camera_intrinsic[1], camera_intrinsic[3]],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    def rectify_stereo_images(self, left_img_path, right_img_path):
        # Load images
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)
        h, w = img_left.shape[:2]

        # Rectify images
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            self.calibs['left']['K'], self.calibs['left']['dist_coeffs'],
            self.calibs['right']['K'], self.calibs['right']['dist_coeffs'],
            (w, h), self.calibs['T'][:3, :3], self.calibs['T'][:3, 3:])

        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.calibs['left']['K'], self.calibs['left']['dist_coeffs'],
            R1, P1, (w, h), cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.calibs['right']['K'], self.calibs['right']['dist_coeffs'],
            R2, P2, (w, h), cv2.CV_32FC1)

        img_left_rectified = cv2.remap(img_left, map1_left, map2_left,
                                       cv2.INTER_LINEAR)
        img_right_rectified = cv2.remap(img_right, map1_right, map2_right,
                                        cv2.INTER_LINEAR)

        return img_left_rectified, img_right_rectified


def visualize_results(rectified_left, rectified_right):
    rectified_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2RGB)
    rectified_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2RGB)

    # Draw the rectified images
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(rectified_left, cmap="gray")
    axes[1].imshow(rectified_right, cmap="gray")
    axes[0].axhline(50)
    axes[1].axhline(50)
    axes[0].axhline(150)
    axes[1].axhline(150)
    axes[0].axhline(250)
    axes[1].axhline(250)
    axes[0].axhline(350)
    axes[1].axhline(350)
    plt.suptitle("Rectified images")
    plt.show()


if __name__ == "__main__":
    calib_file_path = "calibs/zedbag_4x4-camchain.yaml"
    left_img_path = "data/IRVLab/Pool/" \
        "2023-07-18-12-26-58_0_Copy_For_Random_Selection/img_175_left.png"
    right_img_path = "data/IRVLab/Pool/" \
        "2023-07-18-12-26-58_0_Copy_For_Random_Selection/img_175_right.png"

    rectify = Rectification(calib_file_path)
    rectified_left, rectified_right = \
        rectify.rectify_stereo_images(left_img_path, right_img_path)

    # Draw the rectified images
    visualize_results(rectified_left, rectified_right)
