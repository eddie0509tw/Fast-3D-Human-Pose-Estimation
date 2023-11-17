import glob
import os
import json


class LoadMDASData:
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
