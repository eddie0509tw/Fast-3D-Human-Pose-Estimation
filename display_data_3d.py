import yaml
import tqdm
import cv2
import numpy as np
from easydict import EasyDict

from tools.load import load_data


def project_3d_to_2d(pose_3d, P):
    pose_2d = P @ np.vstack((pose_3d.T, np.ones((1, pose_3d.shape[0]))))
    pose_2d = pose_2d.T
    pose_2d[:, :2] /= pose_2d[:, 2:]

    return pose_2d[:, :2]


if __name__ == "__main__":
    with open('configs/mads_3d.yaml', 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    train_dataset, valid_dataset, train_loader, valid_loader \
        = load_data(config)

    for epoch in range(config.TRAIN.EPOCH):

        for i, (image_left, image_right, target_3d,
                target_left, target_right, meta) \
                in enumerate(tqdm.tqdm(train_loader)):
            image_left[:, 0] = image_left[:, 0] * 0.229 + 0.485
            image_left[:, 1] = image_left[:, 1] * 0.224 + 0.456
            image_left[:, 2] = image_left[:, 2] * 0.225 + 0.406
            image_left = image_left * 255.0

            image_right[:, 0] = image_right[:, 0] * 0.229 + 0.485
            image_right[:, 1] = image_right[:, 1] * 0.224 + 0.456
            image_right[:, 2] = image_right[:, 2] * 0.225 + 0.406
            image_right = image_right * 255.0

            target_left = target_left.numpy()
            target_right = target_right.numpy()

            target_3d = target_3d.numpy()
            P_left = meta["P_left"].numpy()
            P_right = meta["P_right"].numpy()

            joints_vis = meta["joints_vis"].numpy()

            for j in range(image_left.shape[0]):
                img_left = image_left[j].numpy().\
                    transpose(1, 2, 0).astype(np.uint8)
                img_right = image_right[j].numpy().\
                    transpose(1, 2, 0).astype(np.uint8)

                display_left = img_left.copy()
                display_right = img_right.copy()

                t_left = project_3d_to_2d(target_3d[j], P_left[j])
                t_right = project_3d_to_2d(target_3d[j], P_right[j])

                vis = joints_vis[j]

                # t_left = target_left[j]
                # t_right = target_right[j]

                for k in range(t_left.shape[0]):
                    if vis[k] == 0:
                        continue
                    if t_left[k, 0] < 0 or t_left[k, 1] < 0:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(
                        display_left,
                        (int(t_left[k, 0]), int(t_left[k, 1])),
                        2, (0, 0, 255), -1)

                for k in range(t_right.shape[0]):
                    if vis[k] == 0:
                        continue
                    if t_left[k, 0] < 0 or t_left[k, 1] < 0:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.circle(
                        display_right,
                        (int(t_right[k, 0]), int(t_right[k, 1])),
                        2, (0, 0, 255), -1)

                display = np.concatenate((display_left, display_right), axis=1)
                cv2.imshow("img", display)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("quit display")
                    exit(1)
