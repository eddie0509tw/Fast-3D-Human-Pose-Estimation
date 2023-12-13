import cv2
import os
import yaml
import tqdm
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib
import PIL.Image as pil
from easydict import EasyDict

from tools.load import LoadMADSData
from tools.common import get_projection_matrix, triangulation
from tools.utils import get_max_preds, project, plot_pose_2d, plot_pose_3d
from models.poseresnet import PoseResNet
from models.metrics import calc_mpjpe

matplotlib.use("Agg")


class BaseLine:
    def __init__(self, config):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = PoseResNet(config)
        self.model = model.to(device)

        # Load the model weights
        weight_path = os.path.join("weights", config.MODEL.NAME, "latest.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            assert False, "Model is not exist in {}".format(weight_path)

        self.model.eval()
        self.device = device

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def inference(self, inputs):
        inputs = self.transform(inputs).unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)

        # Get keypoints locations from heatmap
        preds, _ = get_max_preds(outputs.detach().cpu().numpy())
        preds = preds * 4.0
        preds = preds.astype(np.uint8)

        return preds

    def estimate(self, img_left, img_right, meta):
        # TODO: need to adjust
        pose_3d = np.array(meta['pose_3d'])
        mask = np.isnan(pose_3d)
        pose_3d[mask] = 0
        # set the visibility of joints that have NaN values to 0
        joints_vis = np.ones_like(pose_3d)
        joints_vis[mask] = 0
        joints_vis = np.logical_and.reduce(joints_vis, axis=1,
                                           keepdims=True)
        pose_2d_left, pose_2d_right = project(meta, pose_3d)

        preds_left = self.inference(img_left.copy()).squeeze(0)
        preds_right = self.inference(img_right.copy()).squeeze(0)

        img_2d = plot_pose_2d((pose_2d_left, pose_2d_right),
                              (preds_left, preds_right),
                              (img_left, img_right))
        img_2d = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

        PL = get_projection_matrix(meta['cam_left']['intrinsics'],
                                   meta['cam_left']['rotation'],
                                   meta['cam_left']['translation'])
        PR = get_projection_matrix(meta['cam_right']['intrinsics'],
                                   meta['cam_right']['rotation'],
                                   meta['cam_right']['translation'])

        pred_3ds = triangulation(PL, PR, preds_left, preds_right)
        img_3d = plot_pose_3d(pose_3d, pred_3ds)
        
        pred_2ds = [preds_left, preds_right]

        err = calc_mpjpe(
            pred_2ds, pred_3ds, pose_3d,
            pose_2d_left[:, :2], pose_2d_right[:, :2],
            joints_vis)

        ratio = img_2d.shape[1] / img_3d.shape[1]
        img_3d = cv2.resize(
            img_3d,
            (int(img_3d.shape[1] * ratio), int(img_3d.shape[0] * ratio))
        )

        img = np.vstack((img_2d, img_3d))
        return img, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/mads_2d.yaml",
                        help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    movement = "Sports"
    MADS_loader = LoadMADSData("data/MADS_extract/valid",
                               config.MODEL.IMAGE_SIZE, movement)

    method = BaseLine(config)

    images = []
    error = (0, 0)
    for img_left, img_right, meta in tqdm.tqdm(MADS_loader,
                                               total=len(MADS_loader)):
        pose_img, err = method.estimate(img_left, img_right, meta)
        error = (error[0] + err[0], error[1] + err[1])

        im = pil.fromarray(pose_img)
        images.append(im)

    print("MPJPE2D: ", error[0] / MADS_loader.__len__())
    print("MPJPE3D: ", error[1] / MADS_loader.__len__())
    images[0].save(f'{movement}.gif',
                   save_all=True, append_images=images[1:],
                   optimize=False, duration=40, loop=0)
