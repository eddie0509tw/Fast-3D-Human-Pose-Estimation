import os
import glob
import cv2
import numpy as np
import json
import argparse
import scipy.io


class MADSExtracter:
    def __init__(self, calibs_left_path, calibs_right_path,
                 rectified_left_path, rectified_right_path,
                 rectify_stereo):
        # parse calibration data
        self.calibs = self._parse_calibs(calibs_left_path, calibs_right_path)

        # parse rectification data
        left_rectify = self._parse_rectify(rectified_left_path, "left")
        right_rectify = self._parse_rectify(rectified_right_path, "right")
        self.rectify = {'left': left_rectify, 'right': right_rectify}

        self.rectify_stereo = rectify_stereo

    @staticmethod
    def _parse_calibs(calibs_left_path, calibs_right_path):
        """
        Extract the camera intrinsic matrix, extrinsic matrix,
        and distortion coefficients.

        As the intrinsic matrix of the left camera is modified based on
        rectification, and the intrinsic matrix for both cameras are the same,
        we use the intrinsic matrix of the right camera for both cameras.
        """
        # Load the .mat file
        calibs_left_data = scipy.io.loadmat(calibs_left_path)
        calibs_right_data = scipy.io.loadmat(calibs_right_path)

        # focal length
        fc = calibs_right_data['fc']
        # principal point
        cc = calibs_right_data['cc']
        # skew coefficient
        alpha_c = calibs_right_data['alpha_c']
        # distortion coefficients
        kc = calibs_right_data['kc']

        K = np.array(
            [[fc[0][0], alpha_c[0][0] * fc[0][0], cc[0][0]],
             [0,        fc[1][0],                 cc[1][0]],
             [0,        0,                        1]]
            ).astype(np.float32).reshape(3, 3)

        # rotation and translation vector
        rvec_left, tvec_left = calibs_left_data['om'], calibs_left_data['T']
        rvec_right, tvec_right = \
            calibs_right_data['om_ext'], calibs_right_data['T_ext']

        # correct the rotation vector for left camera
        rvec_left = -rvec_left

        R_left, _ = cv2.Rodrigues(rvec_left)
        T_left = tvec_left.reshape(3, 1)
        R_right, _ = cv2.Rodrigues(rvec_right)
        T_right = tvec_right.reshape(3, 1)

        calibs = {
            'cam_left': {
                "intrinsics": K.tolist(),
                "rotation": R_left.tolist(),
                "translation": T_left.tolist(),
                "distortion_coeffs": kc.tolist()
            },
            'cam_right': {
                "intrinsics": K.tolist(),
                "rotation": R_right.tolist(),
                "translation": T_right.tolist(),
                "distortion_coeffs": kc.tolist()
            }
        }

        return calibs

    @staticmethod
    def _parse_rectify(rectified_path, camera):
        assert camera in ["left", "right"], \
            "camera must be either 'left' or 'right'"

        # Load the .mat file
        data = scipy.io.loadmat(rectified_path)

        rectify = {
            "ind_new": data[f'ind_new_{camera}'][:, 0],
            "ind_1": data[f'ind_1_{camera}'][0] - 1,
            "ind_2": data[f'ind_2_{camera}'][0] - 1,
            "ind_3": data[f'ind_3_{camera}'][0] - 1,
            "ind_4": data[f'ind_4_{camera}'][0] - 1,
            "a1": data[f'a1_{camera}'][0],
            "a2": data[f'a2_{camera}'][0],
            "a3": data[f'a3_{camera}'][0],
            "a4": data[f'a4_{camera}'][0],
        }

        return rectify

    def rectify_calibrated(self, img, camera):
        assert camera in ["left", "right"], \
            "camera must be either 'left' or 'right'"

        info = self.rectify[camera]
        ind_new = info['ind_new']
        ind_1, ind_2, ind_3, ind_4 = \
            info['ind_1'], info['ind_2'], info['ind_3'], info['ind_4']
        a1, a2, a3, a4 = \
            info['a1'], info['a2'], info['a3'], info['a4']

        Im = img.copy()
        h, w, c = Im.shape

        Im = Im.reshape((-1, c), order='F')
        I1 = Im[:, 0]
        I2 = Im[:, 1]
        I3 = Im[:, 2]

        Im_new = np.ones_like(Im) * 144

        Im_new[ind_new, 0] = (
            a1 * I1[ind_1] + a2 * I1[ind_2]
            + a3 * I1[ind_3] + a4 * I1[ind_4]).astype(np.uint8)
        Im_new[ind_new, 1] = (
            a1 * I2[ind_1] + a2 * I2[ind_2]
            + a3 * I2[ind_3] + a4 * I2[ind_4]).astype(np.uint8)
        Im_new[ind_new, 2] = (
            a1 * I3[ind_1] + a2 * I3[ind_2]
            + a3 * I3[ind_3] + a4 * I3[ind_4]).astype(np.uint8)

        Im_new = Im_new.reshape((h, w, c), order='F').copy()

        return Im_new

    def extract(self, video_path, camera, output_dir):
        # Create a directory to save the images
        output_path = os.path.join(output_dir, camera)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Loop through the frames of the video
        frame_count = 0
        while True:
            # Read the next frame
            ret, frame = cap.read()

            # If there are no more frames, break out of the loop
            if not ret:
                break

            if self.rectify_stereo:
                frame = self.rectify_calibrated(frame, camera)

            # Save the frame as an image
            img_path = os.path.join(
                output_path, f"{camera}_{frame_count:04d}.jpg")
            cv2.imwrite(img_path, frame)

            # Increment the frame count
            frame_count += 1

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

    def save_gt_pose(self, gt_pose_path, output_dir):
        # Create a directory to save the ground truths
        output_path = os.path.join(output_dir, "pose")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Load the .mat file
        gt_pose = scipy.io.loadmat(gt_pose_path)['GTpose2'][0]

        for i in range(len(gt_pose)):
            info = {
                'calibs_info': self.calibs,
                'pose_3d': gt_pose[i].tolist()
            }

            # save the converted data as a JSON file
            pose_path = os.path.join(output_path, f"gt_pose_{i:04d}.json")
            with open(pose_path, 'w') as f:
                json.dump(info, f, indent=4, sort_keys=True)

    def process(self, video_left_path, video_right_path, gt_pose_path,
                output_dir):
        # path to save output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save ground truth
        self.save_gt_pose(gt_pose_path, output_dir)

        # convert video to images and save ground truth pose into .json file
        # if groud truth pose is available
        self.extract(video_left_path, "left", output_dir)
        self.extract(video_right_path, "right", output_dir)


def read_file(opt):
    for movement in ["HipHop", "Jazz", "Kata", "Sports", "Taichi"]:
        # left camera calibration data
        calibs_left_path = os.path.join(
            opt.depth_data_path, movement, "Calib_C0_left.mat")
        # right camera calibration data
        calibs_right_path = os.path.join(
            opt.multiview_data_path, movement, "Calib_Cam0.mat")

        # stereo retification coefficients
        rectified_left_path = os.path.join(
            opt.depth_data_path, movement, "rect_calib_left.mat")
        rectified_right_path = os.path.join(
            opt.depth_data_path, movement, "rect_calib_right.mat")

        # videos and ground truth pose
        video_left_path = sorted(glob.glob(os.path.join(
                opt.depth_data_path, movement, "*_Left.avi")))
        video_right_path = sorted(glob.glob(os.path.join(
                opt.depth_data_path, movement, "*_Right.avi")))
        gt_pose_path = sorted(glob.glob(os.path.join(
                opt.depth_data_path, movement, "*_GT.mat")))

        # output directory
        output_dir = os.path.join(opt.output_path, movement)

        # extract data
        convert = MADSExtracter(calibs_left_path, calibs_right_path,
                                rectified_left_path, rectified_right_path,
                                opt.rectify_stereo)

        assert len(video_left_path) == len(video_right_path) == \
            len(gt_pose_path), \
            "Number of videos and ground truth pose must be the same"

        for i in range(len(video_left_path)):
            print(f"Processing {movement} {i+1}/{len(video_left_path)}")
            convert.process(video_left_path[i], video_right_path[i],
                            gt_pose_path[i], os.path.join(output_dir, f"{i}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth_data_path', type=str,
                        default="data/MADS/MADS_depth/depth_data",
                        help='path that store stereo images and ground truth '
                             'pose')
    parser.add_argument('--multiview_data_path', type=str,
                        default="data/MADS/MADS_multiview/multi_view_data",
                        help='path that store multiview camera infos, we only '
                             'need the calibration data for the right camera '
                             'here')
    parser.add_argument('--output_path', type=str,
                        default="data/MADS_training",
                        help='path to save processed output')
    parser.add_argument('--rectify_stereo', action='store_true',
                        help='whether to save rectified stereo images')
    opt = parser.parse_args()
    print(opt)

    read_file(opt)
