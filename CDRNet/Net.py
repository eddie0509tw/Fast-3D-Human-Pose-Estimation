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
                in_dim=2048,
                feat_dims=[256, 256, 256]):
        super(Decoder, self).__init__()
        self.nj = nj
        self.nl = len(feat_dims)
        self.in_dim = in_dim
        self.ct_layers = nn.ModuleList()
        last_dim = None

        for i, hidden_dim in enumerate(feat_dims):
            if i == 0:
                self.ct_layers.append(CTBlock(
                                            self.in_dim,
                                            hidden_dim,
                                            1))
            else:
                self.ct_layers.append(CTBlock(
                                            last_dim,
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

        # latent vector input size is (batch_size, c, 8, 8)

        b, c, h, w = z.size()
        _, h_proj, w_proj = proj_mats.size()

        z_ = z.reshape(b, c//w_proj, w_proj, 1, h, w)
        z_ = z_.permute(0, 1, 4, 5, 2, 3).to(torch.float32)
        proj_mats_ = proj_mats.reshape(b, 1, h_proj, w_proj, 1, 1)
        proj_mats_ = proj_mats_.permute(0, 1, 4, 5, 2, 3).to(torch.float32)

        out = torch.matmul(proj_mats_, z_).permute(0, 1, 4, 5, 2, 3)
        out = out.squeeze(3).reshape(b, -1, h, w).contiguous()

        return out


class Canonical_Fusion(nn.Module):
    def __init__(
                self, in_dim=2048, hid_ch1=300, hid_ch2=400,
                kernel_size=1, stride=1, n_views=2):
      
        super(Canonical_Fusion, self).__init__()
        self.in_ch = in_dim
        self.out_ch = in_dim
        self.hid_ch1 = hid_ch1
        self.hid_ch2 = hid_ch2
        self.n_views = n_views

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.hid_ch1, kernel_size, stride),
            nn.BatchNorm2d(self.hid_ch1),
            nn.ReLU(inplace=True)
        )

        self.ftl_inv = FTL()
        self.ftl = FTL()

        self.conv_layer2 = nn.ModuleList()
        for i in range(self.n_views):
            n_ch = self.n_views * self.hid_ch2 if i == 0 else self.hid_ch2
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
            nn.Conv2d(self.hid_ch1, self.out_ch, kernel_size, stride),

            nn.BatchNorm2d(self.out_ch),

            nn.ReLU(inplace=True)
        )

    def concat_z(self, zs):
        out = torch.empty(0).to(zs[0].device)
        for z in zs:
            out = torch.cat([out, z], axis=1)

        return out

    def forward(self, xs, proj_list, proj_inv_list):
        zs = []
        for x, proj in zip(xs, proj_inv_list):
            x = self.conv_layer1(x)

            z = self.ftl_inv(x, proj)

            zs.append(z)

        f = self.concat_z(zs)
        for conv_layer2 in self.conv_layer2:
            f = conv_layer2(f)

        out = []
        for proj in proj_list:
            z = self.ftl(f, proj)
            z = self.out_layer(z)

            out.append(z)  # list of (batch_size, 2048, 8, 8)

        return out


class CDRNet(nn.Module):
    def __init__(
                self, heatmap_size, n_views=2, nj=19, nl=3, 
                decoder_in_dim=2048, decoder_feat_dim=[256, 256, 256],
                encoder_pretrained=True, fusion_in_dim=2048,
                fusion_hid_ch1=300, fusion_hid_ch2=400):
        super(CDRNet, self).__init__()

        self.heatmap_size = heatmap_size
        self.encoder = Encoder(
                                pretrained=encoder_pretrained)
        self.decoder = Decoder(
                                nj=nj, in_dim=decoder_in_dim,
                                feat_dims=decoder_feat_dim)
        self.CF = Canonical_Fusion(
                                    in_dim=fusion_in_dim,
                                    hid_ch1=fusion_hid_ch1,
                                    hid_ch2=fusion_hid_ch2,
                                    n_views=n_views)
        self.n_views = n_views

    def process_heathap(self, feat):
        b, c, h, w = feat.size()
        x = torch.arange(1, w + 1, 1).to(feat.device)
        y = torch.arange(1, h + 1, 1).to(feat.device)
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
        kps_ = kps.reshape(batch_size, nj, self.n_views, 2, 1).contiguous()
        uv = proj_mats[:, :, :, 2:3].repeat(1, 1, 1, 2, 1) * kps_

        A = (uv - proj_mats[:, :, :, :2])
        A = A.reshape(-1, nj,  2*self.n_views, proj_mats.size(-1)).contiguous()

        AtA = A.permute(0, 1, 3, 2) @ A
        B_inv = AtA - 0.001 * torch.eye(AtA.size(-1)).to(AtA.device)
        B_inv = B_inv.to(torch.float32)

        X = torch.randn((batch_size, nj, 4, 1), dtype=torch.float32)
        X = X.to(B_inv.device)

        for _ in range(n_iters):
            X = torch.linalg.solve(B_inv, X)
            X /= torch.norm(X, dim=1, keepdim=True)

        X = X.squeeze(-1)

        return (X[..., :3].mT / X[..., 3].unsqueeze(-1).mT).mT

    def forward(self, xs, proj_list):
        """
        xs(list): size is (batch_size, 3, 256, 256)
        proj_list(list): (batch_size, 3, 4)
        """
        zs = []
        for i in range(self.n_views):
            z = self.encoder(xs[i])
            zs.append(z)

        proj_inv_list = [
            torch.pinverse(proj) for proj in proj_list]
        
        f_out = self.CF(zs, proj_list, proj_inv_list)

        kps = torch.empty(0).to(xs[0].device)
        projs = torch.empty(0).to(xs[0].device)

        for i in range(self.n_views):
            h = self.decoder(f_out[i])
            kps = torch.cat(
                [kps, self.process_heathap(h).unsqueeze(2)], axis=2)
            proj_ = proj_list[i].unsqueeze(1).repeat(1, kps.size(1), 1, 1)
            projs = torch.cat([projs, proj_.unsqueeze(2)], axis=2)

        pred_3ds = self.SII(kps, projs)
        pred_2ds = [kps[:, :, 0, :].squeeze(2), kps[:, :, 1, :].squeeze(2)]

        return pred_2ds, pred_3ds


if __name__ == '__main__':
    model = CDRNet(heatmap_size=(256, 256))
    xs = [torch.randn(32, 3, 256, 256) for _ in range(2)]
    proj_list = [torch.randn(32, 3, 4) for _ in range(2)]
    pred_2ds, pred_3ds = model(xs, proj_list)
    print(pred_2ds[0].shape)
    print(pred_3ds.shape)
