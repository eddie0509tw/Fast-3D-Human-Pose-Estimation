import cv2
import os
import yaml
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from tools.load import LoadMADSData
from tools.common import project_3d_to_2d, get_projection_matrix
from CDRNet.net import CDRNet


def triangulation(P1, P2, pts1, pts2):
    pts3D = []

    for pt1, pt2 in zip(pts1, pts2):
        pt1_ = np.array([pt1[0], pt1[1], 1])
        pt2_ = np.array([pt2[0], pt2[1], 1])

        pt1_ = np.cross(pt1_, np.identity(pt1_.shape[0]) * -1)
        pt2_ = np.cross(pt2_, np.identity(pt2_.shape[0]) * -1)

        M1 = np.array([pt1[1] * P1[2] - P1[1], P1[0] - pt1[0] * P1[2]])
        M2 = np.array([pt2[1] * P2[2] - P2[1], P2[0] - pt2[0] * P2[2]])
        M = np.vstack((M1, M2))

        e, v = np.linalg.eig(M.T @ M)
        idx = np.argmin(e)
        pt3 = v[:, idx]
        pt3 = (pt3 / pt3[-1])[:3]
        pts3D.append(pt3)

    return np.array(pts3D)


def project(meta):
    K_left = meta['cam_left']['intrinsics']
    R_left = meta['cam_left']['rotation']
    T_left = meta['cam_left']['translation']

    K_right = meta['cam_right']['intrinsics']
    R_right = meta['cam_right']['rotation']
    T_right = meta['cam_right']['translation']

    pose_3d = np.array(meta['pose_3d'])
    pose_2d_left = project_3d_to_2d(pose_3d, K_left, R_left, T_left)
    pose_2d_right = project_3d_to_2d(pose_3d, K_right, R_right, T_right)

    return pose_2d_left, pose_2d_right


def plot_body(ax, points, color, label):
    # Define the connections between the joints
    connections = [
        (0, 1),   # body
        (0, 18),  # head
        (1, 6), (6, 7), (7, 8), (8, 9),  # left leg
        (0, 14), (14, 15), (15, 16), (16, 17),  # left arm
        (1, 2), (2, 3), (3, 4), (4, 5),  # right leg
        (0, 10), (10, 11), (11, 12), (12, 13),  # right arm
    ]

    # Plot the skeleton joints
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=color, marker='o', s=2)

    for connection in connections:
        joint1 = points[connection[0]]
        joint2 = points[connection[1]]
        ax.plot([joint1[0], joint2[0]],
                [joint1[1], joint2[1]],
                [joint1[2], joint2[2]], c=color)

    ax.plot([], [], c=color, label=label)


def visualize_pose_3d(pose_3d, pts3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-600, 600)
    ax.set_ylim3d(-1000, 200)
    ax.set_zlim3d(0, 1500)

    rot = Rotation.from_euler('zyx', np.array([0, 0, 90]),
                              degrees=True).as_matrix()
    pose_3d = (rot @ pose_3d.T).T
    pts3D = (rot @ pts3D.T).T

    plot_body(ax, pose_3d, '#03459c', "ground truth")
    plot_body(ax, pts3D, '#27d128', "estimation")

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Human Skeleton')
    ax.legend()

    # Show the plot
    plt.savefig('pose_3d.png')
    plt.show()


def visualize_pose_2d(gt_joints, pred_joints, imgs):
    def plot_joints(joints, img, color):
        for k in range(joints.shape[0]):
            joint = joints[k]
            cv2.circle(img, (int(joint[0]), int(joint[1])),
                       2, color, -1)

    for gt, pred, img in zip(gt_joints, pred_joints, imgs):
        plot_joints(gt, img, (255, 0, 0))
        plot_joints(pred, img, (0, 255, 0))

    img = np.concatenate(imgs, axis=1)
    cv2.imwrite('pose_2d.png', img)


def numpy2torch(x):
    x = torch.from_numpy(x)
    x = x.type(torch.float32)
    return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        raise TypeError("Expects a PyTorch tensor but get : {}"
                        .format(type(x)))

    return x


class CDRNetInferencer:
    def __init__(self, config):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = CDRNet(config)
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

    def inference(self, img_left, img_right, P_left, P_right):
        img_left = self.transform(img_left).unsqueeze(0)
        img_left = img_left.to(self.device)

        img_right = self.transform(img_right).unsqueeze(0)
        img_right = img_right.to(self.device)

        P_left = numpy2torch(P_left).unsqueeze(0)
        P_left = P_left.to(self.device)

        P_right = numpy2torch(P_right).unsqueeze(0)
        P_right = P_right.to(self.device)

        imgs = [img_left, img_right]
        Ps = [P_left, P_right]

        pred_2ds, pred_3ds = self.model(imgs, Ps)

        for i in range(2):
            pred_2ds[i] = to_cpu(pred_2ds[i].squeeze(0) * 4.0)
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

        pred_2ds, pred_3ds = self.inference(img_left, img_right, PL, PR)

        visualize_pose_2d((pose_2d_left, pose_2d_right),
                          (pred_2ds[0], pred_2ds[1]),
                          (img_left, img_right))

        # pts3D = triangulation(PL, PR, pred_2ds[0], pred_2ds[1])
        visualize_pose_3d(np.array(meta['pose_3d']), pred_3ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/mads_3d.yaml",
                        help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    MADS_loader = LoadMADSData("data/MADS_extract/valid",
                               config.MODEL.IMAGE_SIZE)

    method = CDRNetInferencer(config)

    for img_left, img_right, meta in MADS_loader:
        method.estimate(img_left, img_right, meta)
        exit(1)
