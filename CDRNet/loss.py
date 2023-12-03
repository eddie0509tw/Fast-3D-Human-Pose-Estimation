import torch
import torch.nn as nn


class MPJPELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(MPJPELoss, self).__init__()
        self.use_target_weight = use_target_weight

    def cdist(self, x, y):
        return torch.sqrt((x - y) ** 2 + 1e-15).mean()

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
