import torch
import torch.nn as nn
import torch.nn.functional as F

class MPJPELoss:
    def __init__(self, mode="L2"):
        self.mode = mode

    def __call__(self, pred, target):
        # pred: (batch_size, 19, 3) for 3d or (batch_size, 19, 2) for 2d
        # target: (batch_size, 19, 3) or (batch_size, 19, 2) for 2d
        assert pred.size() == target.size() and len(pred.size()) == 3
        loss = self.calc_loss(pred, target)

        return loss

    def calc_loss(self, pred, target):

        if self.mode == "L2":
            loss = F.mse_loss(pred, target)
        elif self.mode == "L1":
            loss = F.l1_loss(pred, target)
        return loss