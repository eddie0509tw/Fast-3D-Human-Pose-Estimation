import os
import json
import copy
import cv2
import torch
import glob
import numpy as np

from .mads import MADS2DDataset
from tools.common import project_3d_to_2d


class MADS3DDataset(MADS2DDataset):
    def __init__(self, cfg, image_set):
        super().__init__(cfg, image_set)

        self.db = self._get_db()

    def __getitem__(self, idx):
        pass

    def _get_db(self):
        # TODO: You will have to reimplement this function as MADS2DDataset
        #       only loads right images and 2D keypoints for training
        pass

    def preprocess(self, image, joints, joints_vis, c, s, r, origin_size):
        # TODO: You may need to implement this function
        pass
