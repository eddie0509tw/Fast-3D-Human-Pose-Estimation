import cv2
import os
import yaml
import argparse
import torch
import tqdm
import numpy as np
import torchvision.transforms as transforms
import matplotlib
import PIL.Image as pil
from easydict import EasyDict

from tools.load import LoadMADSData
from tools.common import get_projection_matrix
from tools.utils import project, plot_pose_2d, plot_pose_3d, to_cpu, \
                    numpy2torch
from models.cdrnet import CDRNet
from models.metrics import calc_mpjpe

matplotlib.use("Agg")


class CDRNetInferencer:
    def __init__(self, config):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = CDRNet(config)
        self.model = model.to(device)

        # Load the model weights
        weight_path = os.path.join("weights", config.MODEL.NAME, "best.pth")
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

    def inference(self, img_left, img_right, P_left, P_right):
        img_left = self.transform(img_left).unsqueeze(0)
        img_left = img_left.to(self.device)

        img_right = self.transform(img_right).unsqueeze(0)
        img_right = img_right.to(self.device)

        P_left = numpy2torch(P_left[:3]).unsqueeze(0)
        P_left = P_left.to(self.device)

        P_right = numpy2torch(P_right[:3]).unsqueeze(0)
        P_right = P_right.to(self.device)

        imgs = [img_left, img_right]
        Ps = [P_left, P_right]

        pred_2ds, pred_3ds = self.model(imgs, Ps)

        for i in range(2):
            pred_2ds[i] = to_cpu(pred_2ds[i].squeeze(0))
        pred_3ds = to_cpu(pred_3ds.squeeze(0))

        return pred_2ds, pred_3ds

    def estimate(self, img_left, img_right, meta):
        pose_3d = np.array(meta['pose_3d'])
        mask = np.isnan(pose_3d)
        pose_3d[mask] = 0

        # set the visibility of joints that have NaN values to 0
        joints_vis = np.ones_like(pose_3d)
        joints_vis[mask] = 0
        joints_vis = np.logical_and.reduce(joints_vis, axis=1,
                                           keepdims=True)
        pose_2d_left, pose_2d_right = project(meta, pose_3d)

        PL = get_projection_matrix(meta['cam_left']['intrinsics'],
                                   meta['cam_left']['rotation'],
                                   meta['cam_left']['translation'])
        PR = get_projection_matrix(meta['cam_right']['intrinsics'],
                                   meta['cam_right']['rotation'],
                                   meta['cam_right']['translation'])

        pred_2ds, pred_3ds = self.inference(img_left, img_right, PL, PR)

        img_2d = plot_pose_2d((pose_2d_left, pose_2d_right),
                              (pred_2ds[0], pred_2ds[1]),
                              (img_left, img_right))
        img_2d = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

        img_3d = plot_pose_3d(pose_3d, pred_3ds)

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
        cv2.imwrite("test.jpg", img)

        return img, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/mads_3d.yaml",
                        help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    movement = "Taichi"
    MADS_loader = LoadMADSData("data/MADS_extract/valid",
                               config.MODEL.IMAGE_SIZE, movement)

    method = CDRNetInferencer(config)

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
