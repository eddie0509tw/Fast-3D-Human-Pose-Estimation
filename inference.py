import cv2
import os
import argparse
import torch
import tqdm
import glob
import numpy as np
import torchvision.transforms as transforms
import matplotlib
import subprocess

from tools.load import LoadMADSData
from tools.common import get_projection_matrix
from tools.utils import (project, plot_pose_2d, plot_pose_3d, to_cpu,
                         numpy2torch, plot_error)
from models.cdrnet import CDRNet
from models.metrics import mpjpe

matplotlib.use("Agg")


class CDRNetInferencer:
    def __init__(self, num_layers, num_joints, weight_path):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = CDRNet(num_layers, num_joints)
        self.model = model.to(device)

        # Load the model weights
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device)
            model_weights = checkpoint["state_dict"]

            # update keys by dropping `model.`
            for key in list(model_weights):
                weight = model_weights.pop(key)
                model_weights[key.replace("model.", "")] = weight

            model.load_state_dict(model_weights)
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
        pose_2d_left, pose_2d_right = project(meta)

        PL = get_projection_matrix(meta['cam_left']['intrinsics'],
                                   meta['cam_left']['rotation'],
                                   meta['cam_left']['translation'])
        PR = get_projection_matrix(meta['cam_right']['intrinsics'],
                                   meta['cam_right']['rotation'],
                                   meta['cam_right']['translation'])

        target_3d = meta['pose_3d']
        visibility = meta['visibility']

        pred_2ds, pred_3ds = self.inference(img_left, img_right, PL, PR)

        img_2d = plot_pose_2d((pose_2d_left, pose_2d_right),
                              (pred_2ds[0], pred_2ds[1]),
                              (img_left, img_right),
                              visibility)
        img_2d = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

        img_3d = plot_pose_3d(target_3d, pred_3ds, visibility)

        error = mpjpe(pred_3ds, target_3d, visibility)

        return img_2d, img_3d, error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", type=int, default=256,
                        help="height of the input image")
    parser.add_argument("--image_width", type=int, default=256,
                        help="width of the input image")
    parser.add_argument("--num_layers", type=int, default=101,
                        help="Number of layers of the model, i.e. ResNet101")
    parser.add_argument("--num_joints", type=int, default=19,
                        help="Number of joints of the human body")
    parser.add_argument("--weight_path", type=str,
                        default="weights/cdrnet_101_256_mads_3d/last-v4.ckpt",
                        help="Path to the model weights")
    parser.add_argument("--movement", type=str, default="HipHop",
                        help="Name of the movement to be evaluated")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Path to the output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert args.image_height == args.image_width, \
        "image_height and image_width must be the same"

    image_size = [args.image_height, args.image_width]
    MADS_loader = LoadMADSData("data/MADS_extract/valid",
                               image_size, args.movement)

    method = CDRNetInferencer(args.num_layers,
                              args.num_joints,
                              args.weight_path)

    errors = []
    for idx, (img_left, img_right, meta) in enumerate(
            tqdm.tqdm(MADS_loader, total=len(MADS_loader))):
        pose_img_2d, pose_img_3d, error = \
            method.estimate(img_left, img_right, meta)

        errors.append(error)
        error_plot = plot_error(errors, len(MADS_loader))

        pose_img = np.hstack((np.vstack((pose_img_2d, error_plot)),
                              pose_img_3d))
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("pose", pose_img)
        cv2.waitKey(10)
        cv2.imwrite(os.path.join(output_dir, f"{idx:05d}.jpg"), pose_img)
    cv2.destroyAllWindows()

    os.chdir(output_dir)
    subprocess.call([
        'ffmpeg', '-framerate', '30', '-i', '%05d.jpg', '-r', '10',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'copy',
        '-b:v', '3000k',
        args.movement + '.mp4'
    ])
    # remove result images
    for file_name in glob.glob("*.jpg"):
        os.remove(file_name)
