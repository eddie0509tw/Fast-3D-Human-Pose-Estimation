import os
import json
import copy
import cv2
import torch
import glob
import random
import numpy as np

from .mads import MADS2DDataset
from .transforms import get_affine_transform
from tools.common import get_projection_matrix


def numpy2torch(x):
    x = torch.from_numpy(x)
    x = x.type(torch.float32)
    return x


class MADS3DDataset(MADS2DDataset):
    def __init__(self, cfg, image_set):
        super().__init__(cfg, image_set)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        img_left = cv2.imread(db_rec['image_left'], cv2.IMREAD_COLOR)
        img_right = cv2.imread(db_rec['image_right'], cv2.IMREAD_COLOR)

        if img_left is None:
            raise ValueError('Fail to read {}'.format(db_rec['image_left']))
        if img_right is None:
            raise ValueError('Fail to read {}'.format(db_rec['image_right']))

        P_left = db_rec['P_left']
        P_right = db_rec['P_right']

        h, w = img_left.shape[:2]
        s = 1
        r = 0
        c = np.array([w / 2, h / 2])
        origin_size = min(h, w)

        img_left, img_right, P_left, P_right = self.preprocess(
            img_left, img_right, P_left, P_right, c, s, r, origin_size)

        # convert images to torch.tensor and normalize it
        img_left = self.transform(img_left)
        img_right = self.transform(img_right)

        # get ground truth 3D pose
        pose_3d = db_rec['pose_3d']
        target_3d = numpy2torch(pose_3d)

        # get ground truth 2D pose
        target_2d_left = self._project_3d_to_2d(pose_3d, P_left)
        target_2d_right = self._project_3d_to_2d(pose_3d, P_right)

        target_2d_left = numpy2torch(target_2d_left)
        target_2d_right = numpy2torch(target_2d_right)

        joints_vis = numpy2torch(db_rec['joints_vis'])

        # only use the first 3 rows of the projection matrix
        P_left = numpy2torch(P_left[:3])
        P_right = numpy2torch(P_right[:3])

        meta = {
            'image_left': db_rec['image_left'],
            'image_right': db_rec['image_right'],
            'joints_vis': joints_vis,
            'P_left': P_left,
            'P_right': P_right,
            'center': c,
            'scale': s,
            'rotation': r,
        }

        return img_left, img_right, target_3d, \
            target_2d_left, target_2d_right, meta

    def _project_3d_to_2d(self, pose_3d, P):
        pose_2d = P @ np.vstack((pose_3d.T, np.ones((1, pose_3d.shape[0]))))
        pose_2d = pose_2d.T[:, :3]
        pose_2d[:, :2] /= pose_2d[:, 2:]

        return pose_2d[:, :2]

    def _get_db(self):
        left_img_paths = sorted(glob.glob(
            os.path.join(self.root, self.image_set, "**/**/left/*.jpg")))
        right_img_paths = sorted(glob.glob(
            os.path.join(self.root, self.image_set, "**/**/right/*.jpg")))
        gt_pose_paths = sorted(glob.glob(
            os.path.join(self.root, self.image_set, "**/**/pose/*.json")))

        assert len(left_img_paths) == len(right_img_paths) \
            == len(gt_pose_paths), \
            "Number of images and ground truths must match"

        gt_db = []
        for i in range(len(left_img_paths)):
            with open(gt_pose_paths[i], 'r') as f:
                data = json.load(f)

                calibs_info = data['calibs_info']
                pose_3d = np.array(data['pose_3d'])

            # set the value of joints that have NaN values to 0
            mask = np.isnan(pose_3d)
            pose_3d[mask] = 0

            # set the visibility of joints that have NaN values to 0
            joints_vis = np.ones_like(pose_3d)
            joints_vis[mask] = 0
            joints_vis = np.logical_and.reduce(joints_vis, axis=1,
                                               keepdims=True)

            P_left = get_projection_matrix(
                calibs_info['cam_left']['intrinsics'],
                calibs_info['cam_left']['rotation'],
                calibs_info['cam_left']['translation'])

            P_right = get_projection_matrix(
                calibs_info['cam_right']['intrinsics'],
                calibs_info['cam_right']['rotation'],
                calibs_info['cam_right']['translation'])

            # Only use the images from right camera for 2d human pose training
            gt_db.append({
                'image_left': left_img_paths[i],
                'image_right': right_img_paths[i],
                'P_left': P_left,
                'P_right': P_right,
                'joints_vis': joints_vis,
                'pose_3d': pose_3d,
            })

        return gt_db

    def preprocess(self, img_left, img_right, P_left, P_right,
                   c, s, r, origin_size):
        """
        Resize images and joints accordingly for model training.
        If in training stage, random flip, scale, and rotation will be applied.

        Args:
            image: input image
            joints: ground truth keypoints: [num_joints, 3]
            joints_vis: visibility of the keypoints: [num_joints, 3],
                        (1: visible, 0: invisible)
            c: center point of the cropped region
            s: scale factor
            r: degree of rotation
            origin_size: original size of the cropped region
            K: camera projection matrix: [3, 3]
        Returns:
            image, joints, joints_vis, K (after preprocessing)
        """

        if self.image_set == 'train':
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

        trans = get_affine_transform(c, s, r, origin_size, self.image_size)

        img_left = cv2.warpAffine(
            img_left,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        img_right = cv2.warpAffine(
            img_right,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        T = np.eye(4)
        T[:2, :3] = trans
        P_left = T @ P_left
        P_right = T @ P_right

        return img_left, img_right, P_left, P_right
