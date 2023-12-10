import numpy as np

from tools.utils import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.05):
    """Return percentage below threshold while ignoring values with a -1"""
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))

    # extract keypoints from heatmap
    pred, _ = get_max_preds(output)
    target, _ = get_max_preds(target)
    h = output.shape[2]
    w = output.shape[3]
    norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10

    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, pred


def calc_mpjpe(
                pred_2ds,
                pred_3ds,
                gt_3d,
                gt_2d_left,
                gt_2d_right,
                target_weight=None):  
    pred_2d_left = pred_2ds[0]
    pred_2d_right = pred_2ds[1]
    if len(pred_3ds.shape) < 3:
        pred_2d_left = pred_2d_left.reshape(1, -1, 2)
        pred_2d_right = pred_2d_right.reshape(1, -1, 2)
        pred_3ds = pred_3ds.reshape(1, -1, 3)
        gt_3d = gt_3d.reshape(1, -1, 3)
        gt_2d_left = gt_2d_left.reshape(1, -1, 2)
        gt_2d_right = gt_2d_right.reshape(1, -1, 2)

    if target_weight is not None:
        pred_2d_left = pred_2d_left * target_weight
        pred_2d_right = pred_2d_right * target_weight
        pred_3ds = pred_3ds * target_weight
        gt_3d = gt_3d * target_weight
        gt_2d_left = gt_2d_left * target_weight
        gt_2d_right = gt_2d_right * target_weight 

    error_2d_left = np.linalg.norm(pred_2d_left - gt_2d_left, axis=2).mean()
    error_2d_right = np.linalg.norm(pred_2d_right - gt_2d_right, axis=2).mean()

    error_2d = (error_2d_left + error_2d_right) / 2

    error_3d = np.linalg.norm(pred_3ds - gt_3d, axis=2).mean()

    return error_2d, error_3d
