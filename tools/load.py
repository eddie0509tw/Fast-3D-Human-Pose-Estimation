import glob
import os
import json
import torch

from dataset.mpii import MPIIDataset
from dataset.mads import MADS2DDataset


class LoadMADSData:
    def __init__(self, data_path):
        self.metadata = self._gen_metadata(data_path)
        self.frame_idx = list(range(len(self.metadata)))

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

        return self.metadata[idx]

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
        raise NotImplementedError
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
