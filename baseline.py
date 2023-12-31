import cv2
import os
import tqdm
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib
import PIL.Image as pil

from tools.load import LoadMADSData
from tools.common import get_projection_matrix, triangulation
from tools.utils import get_max_preds, project, plot_pose_2d, plot_pose_3d
from models.poseresnet import PoseResNet

matplotlib.use("Agg")


class BaseLine:
    def __init__(self, num_layers, num_joints, weight_path):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = PoseResNet(num_layers, num_joints)
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
        pose_2d_left, pose_2d_right = project(meta)

        preds_left = self.inference(img_left.copy()).squeeze(0)
        preds_right = self.inference(img_right.copy()).squeeze(0)

        img_2d = plot_pose_2d((pose_2d_left, pose_2d_right),
                              (preds_left, preds_right),
                              (img_left, img_right),
                              meta['visibility'])
        img_2d = cv2.cvtColor(img_2d, cv2.COLOR_BGR2RGB)

        PL = get_projection_matrix(meta['cam_left']['intrinsics'],
                                   meta['cam_left']['rotation'],
                                   meta['cam_left']['translation'])
        PR = get_projection_matrix(meta['cam_right']['intrinsics'],
                                   meta['cam_right']['rotation'],
                                   meta['cam_right']['translation'])

        pts3D = triangulation(PL, PR, preds_left, preds_right)
        img_3d = plot_pose_3d(np.array(meta['pose_3d']), pts3D,
                              meta['visibility'])

        ratio = img_2d.shape[1] / img_3d.shape[1]
        img_3d = cv2.resize(
            img_3d,
            (int(img_3d.shape[1] * ratio), int(img_3d.shape[0] * ratio))
        )

        img = np.vstack((img_2d, img_3d))
        return img


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
                        default="weights/poseresnet_101_256_mads_2d/best.ckpt",
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

    method = BaseLine(args.num_layers,
                      args.num_joints,
                      args.weight_path)

    images = []
    for img_left, img_right, meta in tqdm.tqdm(MADS_loader,
                                               total=len(MADS_loader)):
        pose_img = method.estimate(img_left, img_right, meta)

        cv2.imshow("pose", pose_img)
        cv2.waitKey(10)

        im = pil.fromarray(pose_img)
        images.append(im)

    images[0].save('pose.gif',
                   save_all=True, append_images=images[1:],
                   optimize=False, duration=40, loop=0)
