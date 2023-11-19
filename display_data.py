import yaml
import torch
import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict

from dataset.mpii import MPIIDataset


with open('simplebaseline/config.yaml', 'r') as f:
    config = EasyDict(yaml.safe_load(f))


train_dataset = MPIIDataset(config, config.DATASET.TRAIN_SET)
valid_dataset = MPIIDataset(config, config.DATASET.TEST_SET)
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


for epoch in range(config.TRAIN.EPOCH):

    for i, (input, target, target_weight, meta) \
            in enumerate(tqdm.tqdm(train_loader)):
        input[:, 0] = input[:, 0] * 0.229 + 0.485
        input[:, 1] = input[:, 1] * 0.224 + 0.456
        input[:, 2] = input[:, 2] * 0.225 + 0.406
        input = input * 255.0
        target = F.interpolate(
            target, scale_factor=4, mode='bilinear', align_corners=True)

        img = input[0].numpy().transpose(1, 2, 0).astype(np.uint8)
        heatmap = target[0].numpy()

        for i in range(heatmap.shape[0]):
            joint = heatmap[i, :, :]

            joint = cv2.normalize(
                joint, None, alpha=0, beta=255,
                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            joint = cv2.applyColorMap(joint, cv2.COLORMAP_JET)

            display = img * 0.8 + joint * 0.2
            cv2.imshow("img", display.astype(np.uint8))
            key = cv2.waitKey(0)
            if key == ord('q'):
                print("quit display")
                exit(1)
