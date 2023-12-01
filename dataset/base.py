import random
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .transforms import (get_affine_transform, affine_transform,
                         fliplr_joints)


class BaseDataset(Dataset):
    def __init__(self, cfg, image_set):
        self.num_joints = 0
        self.flip_pairs = []
        self.parent_ids = []

        self.root = cfg.DATASET.ROOT
        self.image_set = image_set

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.num_joints = cfg.MODEL.NUM_JOINTS

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.db = self._get_db()

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_db(self):
        raise NotImplementedError

    def preprocess(self, image, joints, joints_vis, c, s, r, origin_size):
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
        Returns:
            image, joints, joints_vis (after preprocessing)
        """

        if self.image_set == 'train':
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                image = image[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, image.shape[1], self.flip_pairs)
                c[0] = image.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, origin_size, self.image_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        return image, joints, joints_vis

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = [i / h for (i, h) in
                               zip(self.image_size, self.heatmap_size)]
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] \
                   or ul[1] >= self.heatmap_size[1] \
                   or br[0] < 0 \
                   or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized,
                # we want the center value to equal 1
                g = np.exp(
                    - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
