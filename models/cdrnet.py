import os
import torch
import torch.nn as nn
from collections import OrderedDict

from .encoder import ResNet
from .decoder import PoseDecoder


class CanonicalFusion(nn.Module):
    def __init__(self, in_dim=2048, hid_ch1=300, hid_ch2=300,
                 n_views=2):

        super(CanonicalFusion, self).__init__()
        out_dim = in_dim

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hid_ch1, kernel_size=1, stride=1),
            nn.BatchNorm2d(hid_ch1),
            nn.ReLU(inplace=True)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(n_views * hid_ch2, hid_ch2, kernel_size=1, stride=1),
            nn.BatchNorm2d(hid_ch2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_ch2, hid_ch2, kernel_size=1, stride=1),
            nn.BatchNorm2d(hid_ch2),
            nn.ReLU(inplace=True)
        )

        self.out_layer = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hid_ch1, out_dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(hid_ch1, out_dim, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )
        ])

    def ftl(self, z, proj_mats):
        # projection matrix input is size (batch_size, 4, 4)
        # latent vector input size is (batch_size, c, 8, 8)

        b, _, h, w = z.size()
        N = proj_mats.size(2)

        z = z.reshape(b, N, -1)
        out = torch.bmm(proj_mats, z)

        out = out.reshape(b, -1, h, w).contiguous()
        return out

    def forward(self, xs, proj_list, proj_inv_list):
        zs = []
        for x, proj in zip(xs, proj_inv_list):
            # map features from 2048 channels to hid_ch1
            x = self.conv_layer1(x)
            # transform feature maps from different views
            # to a shared canonical representation
            z = self.ftl(x, proj)

            zs.append(z)

        # concatenated into a (n × hid_ch1) feature map
        zs = torch.cat(zs, dim=1)
        # process feature map jointly by two 1×1 convolutional layers,
        # producing a unified feature map with hid_ch1 channels that is
        # disentangled from the camera view-point
        f = self.conv_layer2(zs)

        out = []
        for i, proj in enumerate(proj_list):
            # project feature maps back to each original view-point
            z = self.ftl(f, proj)
            # each view- pecific feature map is mapped back to 2048 channels
            z = self.out_layer[i](z)

            out.append(z)  # list of (batch_size, 2048, 8, 8)

        return out


class CDRNet(nn.Module):
    def __init__(
                self, cfg, n_views=2, nj=19, fusion_in_dim=2048,
                fusion_hid_ch1=300, fusion_hid_ch2=400):
        super(CDRNet, self).__init__()

        self.encoder = ResNet(cfg)
        self.CF = CanonicalFusion(in_dim=fusion_in_dim,
                                  hid_ch1=fusion_hid_ch1,
                                  hid_ch2=fusion_hid_ch2,
                                  n_views=n_views)
        self.decoder = PoseDecoder(cfg)
        self.n_views = n_views
        self.nj = nj

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            self.decoder.init_weights()

            # load pretrained model
            checkpoint = torch.load(pretrained)

            # only load weights from encoders
            state_dict = OrderedDict()
            for key in checkpoint.keys():
                if key.startswith('encoder'):
                    state_dict[key] = checkpoint[key]
            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Pretrained model '{}' does not exist."
                             .format(pretrained))

    def process_heathap(self, feat):
        """
        This function computes the 2D location of each joint j by simply
        integrating heatmaps across spatial axes.
        That is, the 2D location of each joint j reqpresents the center of mass
        of the jth feature map.
        Args:
            feat (batch_size, num_joints, N, N): heatmap features
        """

        feat = torch.sigmoid(feat)
        _, _, h, w = feat.size()
        x = torch.arange(1, w + 1, 1).to(feat.device)
        y = torch.arange(1, h + 1, 1).to(feat.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

        cx = torch.sum(grid_x * feat, dim=[2, 3]) / torch.sum(feat, dim=[2, 3])
        cx = (cx - 1).unsqueeze(-1)
        cy = torch.sum(grid_y * feat, dim=[2, 3]) / torch.sum(feat, dim=[2, 3])
        cy = (cy - 1).unsqueeze(-1)

        return torch.cat([cx, cy], dim=-1)

    def dlt(self, proj_matricies, points):
        """
        This function lifts B 2d detections obtained from N viewpoints to 3D
        using the Direct Linear Transform method.
        It computes the eigenvector associated to the smallest eigenvalue
        using Singular Value Decomposition.
        Args:
            proj_matricies torch tensor of shape (B, N, 3, 4):
                sequence of projection matricies
            points torch tensor of of shape (B, N, 2):
                sequence of points'coordinates
        Returns:
            point_3d numpy torch tensor of shape (B, 3): triangulated point
        """

        batch_size = proj_matricies.shape[0]
        n_views = proj_matricies.shape[1]

        A = proj_matricies[:, :, 2:3]\
            .expand(batch_size, n_views, 2, 4) * points.view(-1, n_views, 2, 1)
        A -= proj_matricies[:, :, :2]

        _, _, vh = torch.svd(A.view(batch_size, -1, 4))

        point_3d_homo = -vh[:, :, 3]
        point_3d = (point_3d_homo.transpose(1, 0)[:-1] /
                    point_3d_homo.transpose(1, 0)[-1]).transpose(1, 0)

        return point_3d

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
        img_size = xs[0].size(2)

        zs = []
        for i in range(self.n_views):
            z = self.encoder(xs[i])
            zs.append(z)

        proj_inv_list = [
            torch.linalg.pinv(proj) for proj in proj_list]

        f_out = self.CF(zs, proj_list, proj_inv_list)

        # extract 2D locations of joints from heatmaps
        kps, projs = [], []
        for i in range(self.n_views):
            h = self.decoder(f_out[i])
            heatmap_size = h.size(2)

            kp = self.process_heathap(h)

            # multiply by a factor to scale back to original image size
            kp = kp * (img_size / heatmap_size)

            proj = proj_list[i].unsqueeze(1).repeat(1, self.nj, 1, 1)

            kps.append(kp.unsqueeze(2))
            projs.append(proj.unsqueeze(2))
        kps = torch.cat(kps, dim=2)
        projs = torch.cat(projs, dim=2)

        pred_2ds = [kps[:, :, 0, :].squeeze(2), kps[:, :, 1, :].squeeze(2)]

        # extract 3D locations of joints from Direct Linear Transform
        pred_3ds = []
        for i in range(self.nj):
            kps_3d = self.dlt(projs[:, i, :, :, :], kps[:, i, :, :])
            pred_3ds.append(kps_3d)
        pred_3ds = torch.stack(pred_3ds, axis=1)

        return pred_2ds, pred_3ds


if __name__ == '__main__':
    model = CDRNet()
    xs = [torch.randn(32, 3, 256, 256) for _ in range(2)]
    proj_list = [torch.randn(32, 4, 4) for _ in range(2)]
    pred_2ds, pred_3ds = model(xs, proj_list)
    print(pred_2ds[0].shape)
    print(pred_3ds.shape)
