import torch
import torch.nn as nn


class KeypointsMSELoss(nn.Module):
    def __init__(self):
        super(KeypointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, keypoints_pred, keypoints_gt, weights=None):
        loss = torch.zeros(1, device=keypoints_pred.device)

        batch_size = keypoints_pred.shape[0]
        for i in range(batch_size):
            if weights is not None:
                loss += self.criterion(
                    keypoints_pred[i] * weights[i],
                    keypoints_gt[i] * weights[i]
                )
            else:
                loss += self.criterion(keypoints_pred[i], keypoints_gt[i])

        return loss / batch_size


class KeypointsMSESmoothLoss(nn.Module):
    def __init__(self, threshold=400):
        super().__init__()

        self.threshold = threshold

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        diff = (keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold], 0.1) * (self.threshold ** 0.9)
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        dimension = keypoints_pred.shape[-1]
        loss = torch.sum(torch.abs(keypoints_gt - keypoints_pred) * keypoints_binary_validity)
        loss = loss / (dimension * max(1, torch.sum(keypoints_binary_validity).item()))
        return loss


class KeypointsL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_gt, keypoints_binary_validity):
        loss = torch.sum(torch.sqrt(torch.sum((keypoints_gt - keypoints_pred) ** 2 * keypoints_binary_validity, dim=2)))
        loss = loss / max(1, torch.sum(keypoints_binary_validity).item())
        return loss
