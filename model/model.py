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
        # self.avg_pooling = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.encoder(x)
        # x = self.avg_pooling(x)
        return x


class Decoder(nn.Module):
    def __init__(
                self,
                nj=19,
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


class FTL(nn.Module):
    def __init__(self):
        super(FTL, self).__init__()

    def forward(self, z, proj_mats):
        # projection matrix input is size (batch_size, 3, 4)
        # or (batch_size, 4, 3) for one view

        # latent vector input size is (batch_size, c, 18, 18)

        b, c, h, w = z.size()
        _, h_proj, w_proj = proj_mats.size()

        z_ = z.view(b, c//w_proj, w_proj, 1, h, w)
        proj_mats_ = proj_mats.view(b, 1, h_proj, w_proj, 1, 1)

        out = torch.matmul(proj_mats_, z_).view(b, c // w_proj * h_proj, h, w)
        return out


class Canonical_Fusion(nn.Module):
    def __init__(
                self, in_ch=2048, hid_ch1=300, hid_ch2=400,
                kernel_size=1, stride=1, n_views=2):
      
        super(Canonical_Fusion, self).__init__()
        self.in_ch = in_ch
        self.out_ch = in_ch
        self.hid_ch1 = hid_ch1
        self.hid_ch2 = hid_ch2
        self.n_views = n_views

        self.conv_layer1 = nn.Sequetial(
            nn.Conv2d(self.in_ch, self.hid_ch1, kernel_size, stride),
            nn.BatchNorm2d(self.hid_ch1),
            nn.ReLU(inplace=True)
        )

        self.ftl_inv = FTL()
        self.ftl = FTL()

        self.conv_layer2 = nn.ModuleList()
        for i in range(self.n_views):
            n_ch = self.n_views * self.hid_ch1 if i == 0 else self.hid_ch2
            self.conv_layer2.add_module(
                "CF_conv%d" % i,
                nn.Sequential(
                        nn.Conv2d(
                            n_ch, self.hid_ch2,
                            kernel_size, stride),

                        nn.BatchNorm2d(self.hid_ch2),

                        nn.ReLU(inplace=True)
                ),
            )

        self.out_layer = nn.Sequential(
            nn.Conv2d(self.hid_ch2, self.out_ch, kernel_size, stride),

            nn.BatchNorm2d(self.out_ch),

            nn.ReLU(inplace=True)
        )

    def concat_z(self, zs):
        out = torch.empty(0).to(zs[0].device)
        for z in zs:
            torch.cat([out, z], axis=1)

        return out

    def forward(self, xs, proj_list):
        zs = []
        for x, proj in zip(xs, proj_list):
            x = self.conv_layer1(x)

            z = self.ftl_inv(x, proj)

            zs.append(z)

        f = self.concat_z(zs)
        f = self.conv_layer2(f)

        out = []
        for proj in proj_list:
            z = self.ftl(f, proj)

            z = self.out_layer(z)

            out.append(z)

        return out


class CDRNet(nn.Module):
    def __init__(
                self, heatmap_size, n_views=2):    
        super(CDRNet, self).__init__()

        self.heatmap_size = heatmap_size
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.CF = Canonical_Fusion()
        self.n_views = n_views

    def process_heathap(self, feat):
        b, c, h, w = feat.size()
        x = torch.arange(1, w + 1, 1)
        y = torch.arange(1, h + 1, 1)
        grid_x, grid_y = torch.meshgrid(x, y)

        cx = torch.sum(grid_x * feat, dim=[2, 3]) / torch.sum(feat, dim=[2, 3])
        cx = (cx - 1).unsqueeze(-1)
        cy = torch.sum(grid_y * feat, dim=[2, 3]) / torch.sum(feat, dim=[2, 3])
        cy = (cy - 1).unsqueeze(-1)
      
        return torch.cat([cx, cy], dim=-1)

    def SII(self, kps, proj_mats, n_iters=5):
        """
        kps: (batch_size, nj, n_views, 2)
        proj_mats: (batch_size, nj, n_views, 3, 4)
        """
        batch_size, nj, _, _ = kps.size()
        kps_ = kps.view(batch_size, nj, self.n_views, 2, 1)
        uv = proj_mats[:, :, :, 2:3].repeat(1, 1, 1, 2, 1) * kps_
        A = uv - proj_mats[:, :, :, :2].view(
                                            -1, 2 * self.n_views,
                                            proj_mats.size(-1))
      
        AtA = A.permute(0, 2, 1) @ A
        B_inv = AtA - 0.001 * torch.eye(AtA.size(-1), device=A.device)

        X = torch.randn(
                        (batch_size, 4, 1),
                        mean=0.5, std=0.5,
                        dtype=torch.float32)
       
        for _ in range(n_iters):

            X = torch.linalg.solve(B_inv, X)

            X /= torch.norm(X, dim=1, keepdim=True)

        X = X.squeeze(-1)

        return (X[..., :3].T / X[..., 3].unsqueeze(-1).T).T
   
    def forward(self):
        pass

if __name__ == '__main__':
    model = Encoder()
    print(model)
    out = model(torch.randn(1, 3, 256, 256))
    print(out.shape)
