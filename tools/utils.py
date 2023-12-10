import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation
from .common import project_3d_to_2d
import torch
import cv2
import os
matplotlib.use("Agg")


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def project(meta, pose_3d):
    K_left = meta['cam_left']['intrinsics']
    R_left = meta['cam_left']['rotation']
    T_left = meta['cam_left']['translation']

    K_right = meta['cam_right']['intrinsics']
    R_right = meta['cam_right']['rotation']
    T_right = meta['cam_right']['translation']

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


def plot_pose_3d(pose_3d, pts3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1000, 1000)
    ax.set_ylim3d(-1500, 1500)
    ax.set_zlim3d(0, 1700)

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

    # Convert the plot to numpy array
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)
    plt.close()

    return image_array


def plot_pose_2d(gt_joints, pred_joints, imgs):
    def plot_joints(joints, img, color):
        for k in range(joints.shape[0]):
            joint = joints[k]
            # Check if the joint has NaN values
            if not np.isnan(joint[0]) and not np.isnan(joint[1]):      
                # Plot the joint as a circle
                cv2.circle(img, (int(joint[0]), int(joint[1])),
                        2, color, -1)
                
    for gt, pred, img in zip(gt_joints, pred_joints, imgs):
        plot_joints(gt, img, (255, 0, 0))
        plot_joints(pred, img, (0, 255, 0))

    img = np.concatenate(imgs, axis=1)

    return img


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


def plot_loss(losses, save_path, title):
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    epochs = np.arange(len(losses))
    plt.plot(epochs, np.array(losses))
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(f"{title} vs Epoch")
    save_name = f"{title}.png"
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path)
    plt.show()
    plt.close()
