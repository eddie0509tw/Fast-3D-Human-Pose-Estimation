import yaml
import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
from easydict import EasyDict

from tools.load import load_data
from simplebaseline.utils import get_max_preds


if __name__ == "__main__":
    with open('configs/mads_2d.yaml', 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    train_dataset, valid_dataset, train_loader, valid_loader \
        = load_data(config)

    for epoch in range(config.TRAIN.EPOCH):

        for i, (input, target, target_weight, meta) \
                in enumerate(tqdm.tqdm(train_loader)):
            input[:, 0] = input[:, 0] * 0.229 + 0.485
            input[:, 1] = input[:, 1] * 0.224 + 0.456
            input[:, 2] = input[:, 2] * 0.225 + 0.406
            input = input * 255.0
            preds, _ = get_max_preds(target.numpy())
            target = F.interpolate(
                target, scale_factor=4, mode='bilinear', align_corners=True)
            joints = meta['joints']

            for j in range(input.shape[0]):
                img = input[j].numpy().transpose(1, 2, 0).astype(np.uint8)
                heatmap = target[j].numpy()
                joint = joints[j].numpy()

                pred = preds[j] * 4
                print(pred, joint)

                img = img.copy()

                for k in range(joint.shape[0]):
                    cv2.circle(
                        img, (int(joint[k, 0]), int(joint[k, 1])),
                        2, (0, 255, 0), -1)

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
