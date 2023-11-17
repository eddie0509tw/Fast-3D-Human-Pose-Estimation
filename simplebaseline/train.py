import yaml
import torch
from easydict import EasyDict

from poseresnet import PoseResNet


with open('simplebaseline/config.yaml', 'r') as f:
    data = EasyDict(yaml.safe_load(f))

    model = PoseResNet(data)

    image = torch.rand(1, 3, 384, 384)
    out = model(image)
    print(out.shape)
