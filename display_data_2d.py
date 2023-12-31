import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig

from tools.load import HumanPoseDataModule


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def run(cfg: DictConfig):

    dm = HumanPoseDataModule(
        cfg.dataset,
        cfg.load.num_workers,
        cfg.train.batch_size,
        cfg.train.img_size,
        cfg.target.heatmap_size,
        cfg.target.sigma)
    dm.setup()
    train_loader = dm.train_dataloader()

    for i, (input, target, target_weight, meta) \
            in enumerate(tqdm.tqdm(train_loader)):
        input[:, 0] = input[:, 0] * 0.229 + 0.485
        input[:, 1] = input[:, 1] * 0.224 + 0.456
        input[:, 2] = input[:, 2] * 0.225 + 0.406
        input = input * 255.0

        target = F.interpolate(
            target, scale_factor=4, mode='bilinear', align_corners=True)
        joints = meta['joints']

        for j in range(input.shape[0]):
            img = input[j].numpy().transpose(1, 2, 0).astype(np.uint8)
            heatmap = target[j].numpy()
            joint = joints[j].numpy()

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


if __name__ == "__main__":
    run()
