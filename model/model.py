import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, pretrained=True, output_size=(18, 18)):
        super(Encoder, self).__init__()
        self.pretrained = pretrained

        self.model = models.resnet152(pretrained=self.pretrained)

        # Ignore the FC layer and adjust the pooling layer
        self.encoder = torch.nn.Sequential(*list(self.model.children())[:-2])
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pooling(x)
        return x


class Decoder(nn.Module):
    class Encoder(nn.Module):
        def __init__(self,
                     nj=15,
                     nl=3,
                     in_dim=2048,
                     feat_dims=[256, 256, 256]):
            super(Encoder, self).__init__()
            assert nl == len(feat_dims)

            self.nj = nj
            self.nl = nl
            self.in_dim = in_dim
            self.ct_layers = nn.ModuleList()
            last_dim = None

            for i, hidden_dim in enumerate(feat_dims):
                if i == 0:
                    self.ct_layers.append(CTBlock(self.in_dim,
                                                  hidden_dim,
                                                  1))
                else:
                    self.ct_layers.append(CTBlock(last_dim,
                                                  hidden_dim,
                                                  1))

                last_dim = hidden_dim
            self.conv_last = nn.Conv2d(last_dim, self.nj, 1)

    def forward(self, x):
        for i in range(self.nl):
            x = self.ct_layers[i](x)
        x = self.conv_last(x)
        return x


class CTBlock(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel_size=4,
                 stride=2,
                 padding=1):
        super(CTBlock, self).__init__()

        self.ct = nn.ConvTranspose2d(in_ch, out_ch,
                                     kernel_size,
                                     stride,
                                     padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.ct(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# class FTL(nn.Module):
#     def __init__(self):
#         super(FTL, self).__init__()

#     def forward(self, z, proj_mats):
#         # projection matrix input is size (batch_size, 3, 4)
#         # latent vector input size is (batch_size, c, 18, 18)

#         b, c, h, w = z.size()

#         z_ = z.view()
#         p = proj_mats.view(b, 3, 4)


if __name__ == '__main__':
    model = Encoder()
    print(model)
    out = model(torch.randn(1, 3, 256, 256))
    print(out.shape)