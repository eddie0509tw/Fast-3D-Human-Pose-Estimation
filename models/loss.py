import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for i in range(num_joints):
            heatmap_pred = heatmaps_pred[i].squeeze()
            heatmap_gt = heatmaps_gt[i].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred * target_weight[:, i],
                    heatmap_gt * target_weight[:, i]
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsMSESmoothLoss(nn.Module):
    def __init__(self, use_target_weight, threshold=500):
        super(JointsMSESmoothLoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.threshold = threshold

    def cdist(self, x, y):
        diff = torch.sum((x - y) ** 2, dim=1)
        diff[diff > self.threshold] = \
            torch.pow(diff[diff > self.threshold], 0.1) \
            * (self.threshold ** 0.9)
        return diff.mean()

    def forward(self, output, target, target_weight):
        loss = torch.zeros(1, device=output.device)

        num_joints = output.size(1)

        output = output.split(1, 1)
        target = target.split(1, 1)

        for i in range(num_joints):
            out = output[i].squeeze(1)
            tar = target[i].squeeze(1)

            if self.use_target_weight:
                loss += self.cdist(
                    out * target_weight[:, i],
                    tar * target_weight[:, i]
                )
            else:
                loss += self.cdist(out, tar)

        return loss / num_joints


class MPJPELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(MPJPELoss, self).__init__()
        self.use_target_weight = use_target_weight

    def cdist(self, x, y):
        return torch.sqrt(torch.sum((x - y) ** 2, dim=1) + 1e-15).mean()

    def forward(self, output, target, target_weight):
        loss = torch.zeros(1, device=output.device)

        num_joints = output.size(1)

        output = output.split(1, 1)
        target = target.split(1, 1)

        for i in range(num_joints):
            out = output[i].squeeze(1)
            tar = target[i].squeeze(1)

            if self.use_target_weight:
                loss += self.cdist(
                    out * target_weight[:, i],
                    tar * target_weight[:, i]
                )
            else:
                loss += self.cdist(out, tar)

        return loss / num_joints
