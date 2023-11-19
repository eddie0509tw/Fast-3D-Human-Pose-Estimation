from torch import nn

from .encoder import ResNet
from .decoder import PoseDecoder


class PoseResNet(nn.Module):
    def __init__(self, cfg):
        super(PoseResNet, self).__init__()

        self.encoder = ResNet(cfg)
        self.decoder = PoseDecoder(cfg)

    def forward(self, x):
        features = self.encoder(x)
        heatmaps = self.decoder(features)

        return heatmaps
