import os
import glob
import cv2
import numpy as np
import json
import scipy.io


class ImageConverter:
    def __init__(self, calibs_path, rectified_left_path, rectified_right_path):
        # Load the .mat file
        calibs_data = scipy.io.loadmat(calibs_path)
        self.calibs = self._parse_calibs(calibs_data,
                                         Kata=("Kata" in calibs_path))
        print("Camera Info:", self.calibs)

        left_rectify = self._parse_rectify(rectified_left_path, "left")
        right_rectify = self._parse_rectify(rectified_right_path, "right")
        self.rectify = {'left': left_rectify, 'right': right_rectify}

    @staticmethod
    def _parse_calibs(calibs_data, Kata=False):
        """
        Extract the camera intrinsic matrix, extrinsic matrix,
        and distortion coefficients
        """
        # focal length
        fc = calibs_data['fc']
        # principal point
        cc = calibs_data['cc']
        # skew coefficient
        alpha_c = calibs_data['alpha_c']
        # distortion coefficients
        kc = calibs_data['kc']
        # rotation vector
        rvec = calibs_data['om']
        # translation vector
        tvec = calibs_data['T']

        # correct the rotation vector if the movement is "Kata"
        if Kata:
            rvec = -rvec

        K = np.array(
            [[fc[0][0], alpha_c[0][0], cc[0][0]],
             [0, fc[1][0], cc[1][0]],
             [0, 0, 1]]
            ).astype(np.float32).reshape(3, 3)

        R, _ = cv2.Rodrigues(rvec)
        T = tvec.reshape(3, 1)
        dist = kc.reshape(5, 1)

        calibs = {
            "K": K,
            "R": R,
            "T": T,
            "dist_coeffs": dist
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

    @staticmethod
    def _parse_gt_pose(gt_pose_path):
        # Load the .mat file
        gt_pose = scipy.io.loadmat(gt_pose_path)['GTpose2'][0]
        return gt_pose

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

    def world_to_camera(self, points):
        # Construct the transformation matrix
        Rt = np.concatenate((self.calibs["R"], self.calibs["T"]), axis=1)
        Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)

        # Transform the 3D points into the camera coordinate system
        points_hom = np.vstack((points.T,
                                np.ones((1, points.shape[0]))))

        points_hom = Rt @ points_hom

        return points_hom[:3].T

    def camera_to_image(self, points):
        points_2d = self.calibs["K"] @ points.T

        points_2d = points_2d.T
        points_2d[:, :2] /= points_2d[:, 2:]

        return points_2d

    def convert2img(self, video_path, camera, output_dir, gt_pose=None):
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

            frame = self.rectify_calibrated(frame, camera)

            if gt_pose is not None:
                pose = np.array(gt_pose[frame_count])
                pose = self.world_to_camera(pose)
                pose = self.camera_to_image(pose)

                for point in pose:
                    # avoid ploting keypoints with NaN values
                    if np.any(np.isnan(point)):
                        continue
                    cv2.circle(frame,
                               (int(point[0]), int(point[1])),
                               3, (0, 255, 0), -1)

            # Save the frame as an image
            output_path = os.path.join(output_dir, f"{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)

            # Increment the frame count
            frame_count += 1

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

    def savepose(self, gt_pose, output_dir):
        # create an empty dictionary to store the converted data
        gt_pose_dict = {}

        # loop through each pose in gt_pose
        for i, pose in enumerate(gt_pose):
            # convert the pose from world coordinate to camera coordinate
            pose = self.world_to_camera(np.array(pose))
            # add the pose to the dictionary with a key of the current index
            gt_pose_dict[i] = pose.tolist()

        info = {
            'cam_info': {
                'intrinsic_matrix': self.calibs['K'].tolist(),
                'distortion_coefficients': self.calibs['dist_coeffs'].tolist(),
            },
            'pose': gt_pose_dict
        }

        # save the converted data as a JSON file
        with open(os.path.join(output_dir, 'gt.json'), 'w') as f:
            json.dump(info, f, indent=4, sort_keys=True)

    def process(self, video_left_path, video_right_path, gt_pose_path,
                output_dir):
        # Create a directory to save the images
        left_output_dir = os.path.join(output_dir, "left")
        if not os.path.exists(left_output_dir):
            os.makedirs(left_output_dir)

        right_output_dir = os.path.join(output_dir, "right")
        if not os.path.exists(right_output_dir):
            os.makedirs(right_output_dir)

        # read ground truth pose
        gt_pose = self._parse_gt_pose(gt_pose_path)

        # convert video to images
        self.convert2img(video_left_path, "left", left_output_dir, gt_pose)
        self.convert2img(video_right_path, "right", right_output_dir, gt_pose)

        # save ground truth pose into .json file
        self.savepose(gt_pose, output_dir)


def read_file(data_path, movements, output_root):
    for movement in movements:
        # camera calibration and stereo retification coefficients
        calibs_path = os.path.join(data_path, movement, "Calib_C0_left.mat")
        rectified_left_path = os.path.join(
            data_path, movement, "rect_calib_left.mat")
        rectified_right_path = os.path.join(
            data_path, movement, "rect_calib_right.mat")

        # videos and ground truth pose
        video_left_path = sorted(
            glob.glob(os.path.join(data_path, movement, "*_Left.avi")))
        video_right_path = sorted(
            glob.glob(os.path.join(data_path, movement, "*_Right.avi")))
        gt_pose_path = sorted(
            glob.glob(os.path.join(data_path, movement, "*_GT.mat")))

        # output directory
        output_dir = os.path.join(output_root, movement)

        # extract data
        convert = ImageConverter(calibs_path, rectified_left_path,
                                 rectified_right_path)

        assert len(video_left_path) == len(video_right_path) == \
            len(gt_pose_path), \
            "Number of videos and ground truth pose must be the same"

        for i in range(len(video_left_path)):
            print(f"Processing {movement} {i+1}/{len(video_left_path)}")
            convert.process(video_left_path[i], video_right_path[i],
                            gt_pose_path[i], os.path.join(output_dir, f"{i}"))


if __name__ == "__main__":
    data_path = "data/MADS/MADS_depth/depth_data"
    output_root = "output"
    movements = ["HipHop", "Jazz", "Kata", "Sports", "Taichi"]
    read_file(data_path, movements, output_root)
