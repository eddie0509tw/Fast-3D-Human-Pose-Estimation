import os
import json
import copy
import cv2
import torch
import glob
import numpy as np

from .base import BaseDataset
from tools.common import project_3d_to_2d


class MADS2DDataset(BaseDataset):
    def __init__(self, cfg, image_set):
        super().__init__(cfg, image_set)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.flip_pairs = [[2, 6], [3, 7], [4, 8], [5, 9], [10, 14],
                           [11, 15], [12, 16], [13, 17]]
        self.parent_ids = [0, 0, 1, 2, 3, 4, 1, 6, 7, 8, 0, 10, 11,
                           12, 0, 14, 15, 16, 0]

        self.db = self._get_db()

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR)

        if data_numpy is None:
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints']
        joints_vis = db_rec['joints_vis']

        h, w = data_numpy.shape[:2]
        s = 1
        r = 0
        c = np.array([w / 2, h / 2])
        origin_size = min(h, w)

        image, joints, joints_vis = self.preprocess(
            data_numpy, joints, joints_vis, c, s, r, origin_size)

        # convert images to torch.tensor and normalize it
        image = self.transform(image)

        # convert 2d keypoints to heatmaps
        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'K': db_rec['K'],
            'R': db_rec['R'],
            'T': db_rec['T']
        }

        return image, target, target_weight, meta

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

            K = np.array(
                    calibs_info['cam_right']['intrinsics'])
            R = np.array(
                    calibs_info['cam_right']['rotation'])
            T = np.array(
                    calibs_info['cam_right']['translation'])

            # set the value of joints that have NaN values to 0
            mask = np.isnan(pose_3d)
            pose_3d[mask] = 0

            # set the visibility of joints that have NaN values to 0
            joints_vis = np.ones_like(pose_3d)
            joints_vis[mask] = 0

            pose_2d = project_3d_to_2d(pose_3d, K, R, T)

            # Only use the images from right camera for 2d human pose training
            gt_db.append({
                'image': right_img_paths[i],
                'joints': pose_2d,
                'joints_vis': joints_vis,
                'K': K,
                'R': R,
                'T': T
            })

        return gt_db
