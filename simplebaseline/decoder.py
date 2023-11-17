from torch import nn


class PoseDecoder(nn.Module):
    def __init__(self, cfg):
        super(PoseDecoder, self).__init__()

        self.deconv1 = self._make_deconv_layer(
            2048, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = self._make_deconv_layer(
            256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = self._make_deconv_layer(
            256, 256, kernel_size=4, stride=2, padding=1,)

        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, c_in, c_out, kernel_size, stride, padding):
        layers = []
        layers.append(
            nn.ConvTranspose2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=0,
                bias=False))
        layers.append(nn.BatchNorm2d(c_out, momentum=0.1))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        x = self.final_layer(x)

        return x
