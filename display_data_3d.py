import yaml
import tqdm
import cv2
import numpy as np
from easydict import EasyDict

from tools.load import load_data


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

            for j in range(image_left.shape[0]):
                img_left = image_left[j].numpy().\
                    transpose(1, 2, 0).astype(np.uint8)
                img_right = image_right[j].numpy().\
                    transpose(1, 2, 0).astype(np.uint8)

                display_left = img_left.copy()
                display_right = img_right.copy()

                t_left = target_left[j]
                t_right = target_right[j]

                for k in range(t_left.shape[0]):
                    cv2.circle(
                        display_left,
                        (int(t_left[k, 0]), int(t_left[k, 1])),
                        2, (0, 0, 255), -1)

                for k in range(t_right.shape[0]):
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
