import glob
import os
import copy
import json
import cv2
import torch
import numpy as np

from dataset.mpii import MPIIDataset
from dataset.mads import MADS2DDataset
from dataset.mads_3d import MADS3DDataset
from dataset.transforms import get_affine_transform


class LoadMADSData:
    def __init__(self, data_path, image_size):
        self.metadata = self._gen_metadata(data_path)
        self.frame_idx = list(range(len(self.metadata)))

        self.image_size = image_size

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.metadata)

    def __next__(self):
        if self.count == len(self.frame_idx):
            raise StopIteration
        idx = self.frame_idx[self.count]
        self.count += 1

        meta = copy.deepcopy(self.metadata[idx])

        left_img = cv2.imread(meta['left_img_path'], cv2.IMREAD_COLOR)
        right_img = cv2.imread(meta['right_img_path'], cv2.IMREAD_COLOR)

        h, w = left_img.shape[:2]
        c = np.array([w / 2, h / 2])
        origin_size = min(h, w)

        trans = get_affine_transform(c, 1, 0, origin_size, self.image_size)

        # crop and resize images to match the model input size
        left_img = cv2.warpAffine(
            left_img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        right_img = cv2.warpAffine(
            right_img,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        # correct intrinsics matrices according to cropped and resized results
        K_left = meta['cam_left']['intrinsics']
        K_right = meta['cam_right']['intrinsics']

        K_left = np.vstack((trans @ K_left, np.array([0, 0, 1])))
        K_right = np.vstack((trans @ K_right, np.array([0, 0, 1])))

        meta['cam_left']['intrinsics'] = K_left
        meta['cam_right']['intrinsics'] = K_right

        return left_img, right_img, meta

    def _gen_metadata(self, data_path):
        left_img_paths = sorted(glob.glob(
            os.path.join(data_path, "**/**/left/*.jpg")))
        right_img_paths = sorted(glob.glob(
            os.path.join(data_path, "**/**/right/*.jpg")))
        gt_pose_paths = sorted(glob.glob(
            os.path.join(data_path, "**/**/pose/*.json")))

        assert len(left_img_paths) == len(right_img_paths) \
            == len(gt_pose_paths), \
            "Number of images and ground truths must match"

        metadata = []
        for i in range(len(left_img_paths)):
            with open(gt_pose_paths[i], 'r') as f:
                data = json.load(f)

                calibs_info = data['calibs_info']
                pose_3d = data['pose_3d']

            metadata.append({
                'cam_left': calibs_info['cam_left'],
                'cam_right': calibs_info['cam_right'],
                'left_img_path': left_img_paths[i],
                'right_img_path': right_img_paths[i],
                'pose_3d': pose_3d
            })

        return metadata


def load_data(config):
    if config.DATASET.TYPE == "MPII":
        train_dataset = MPIIDataset(config, config.DATASET.TRAIN_SET)
        valid_dataset = MPIIDataset(config, config.DATASET.TEST_SET)
    elif config.DATASET.TYPE == "MADS_2d":
        train_dataset = MADS2DDataset(config, config.DATASET.TRAIN_SET)
        valid_dataset = MADS2DDataset(config, config.DATASET.TEST_SET)
    elif config.DATASET.TYPE == "MADS_3d":
        train_dataset = MADS3DDataset(config, config.DATASET.TRAIN_SET)
        valid_dataset = MADS3DDataset(config, config.DATASET.TEST_SET)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    return train_dataset, valid_dataset, train_loader, valid_loader
