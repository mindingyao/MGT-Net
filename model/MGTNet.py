import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_50 import Spatial
from .temporal_50 import Temporal
from transformer.MGTrans import VisionTransformer as MGTrans
from transformer.MGTrans import CONFIGS as CONFIGS_ViT_seg
from options import config
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.ReLU(True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)

    def forward(self, img):
        f = self.fc2(self.relu1(self.fc1(self.avg_pool(img))))  # B 2C 1 1
        return img.mul(self.sigmoid(f)) + img


class FPM(nn.Module):
    def __init__(self, in_planes, ratio):
        super(FPM, self).__init__()
        self.se1 = ChannelAttention(in_planes, ratio)
        self.se2 = ChannelAttention(in_planes, ratio)

        self.gate = nn.Sequential(nn.Conv2d(in_planes, 2, kernel_size=1), nn.Sigmoid())

    def forward(self, rgb, flow):
        rgb_ = self.se1(rgb)
        flow_ = self.se2(flow)

        fusion = rgb_ * flow_ + rgb_ + flow_

        gate = self.gate(fusion)

        return rgb.mul(gate[:, :1, :, :]) + rgb, flow.mul(gate[:, 1:, :, :]) + flow


class Interactive(nn.Module):
    def __init__(self, spatial_ckpt=None, temporal_ckpt=None, pretrained=False):
        super(Interactive, self).__init__()
        self.spatial_net = Spatial(pretrained)
        self.temporal_net = Temporal(pretrained)
        if spatial_ckpt is not None:
            self.spatial_net.load_state_dict(torch.load(spatial_ckpt)['state_dict'])
            print("Successfully load spatial:{}".format(spatial_ckpt))
        if temporal_ckpt is not None:
            self.temporal_net.load_state_dict(torch.load(temporal_ckpt)['state_dict'])
            print("Successfully load temporal:{}".format(temporal_ckpt))

        self.config = config
        config_vit = CONFIGS_ViT_seg[config.vit_name]

        self.net = MGTrans(config_vit, img_size=config.img_size).cuda()
        if pretrained:
            self.net.load_from(weights=np.load(config_vit.pretrained_path))

        self.fpm1 = FPM(256, 16)
        self.fpm2 = FPM(512, 16)
        self.fpm3 = FPM(1024, 16)
        self.fpm4 = FPM(2048, 16)
        self.fpm5 = FPM(256, 16)

        self.T_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer3_s = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer3_t = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer4_s = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer4_t = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.T_layer5_s = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=True),  # 默认是输入：96+96
            nn.ReLU(inplace=True)
        )

        self.T_layer5_t = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.squeeze_3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.squeeze_4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.squeeze_5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(320, 320, 3, 1, 1),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, 320, 3, 1, 1),
            nn.BatchNorm2d(320),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.blockaspp = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.pred_head1 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        self.pred_head2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        self.pred_head3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        self.pred_head4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        self.pred_head5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

    def encoder(self, x, flow):
        B, C, H, W = x.size()
        spatial0 = self.spatial_net.inconv(x.view(B, C, H, W))
        temporal0 = self.temporal_net.inconv(flow.view(B, C, H, W))
        _, C, H, W = spatial0.size()

        spatial1 = self.spatial_net.encoder1(spatial0)
        temporal1 = self.temporal_net.encoder1(temporal0)

        S1_1T, T1_1S = self.fpm1(spatial1, temporal1)

        h1_spatial = S1_1T

        h1_temporal = T1_1S

        spatial2 = self.spatial_net.encoder2(spatial1)
        temporal2 = self.temporal_net.encoder2(temporal1)

        S2_1T, T2_1S = self.fpm2(spatial2, temporal2)

        h2_spatial = S2_1T

        h2_temporal = T2_1S

        spatial3 = self.spatial_net.encoder3(spatial2)
        temporal3 = self.temporal_net.encoder3(temporal2)

        S3_1T, T3_1S = self.fpm3(spatial3, temporal3)

        h3_spatial = S3_1T
        h3_temporal = T3_1S

        spatial4 = self.spatial_net.encoder4(spatial3)
        temporal4 = self.temporal_net.encoder4(temporal3)

        S4_1T, T4_1S = self.fpm4(spatial4, temporal4)

        h4_spatial = S4_1T

        h4_temporal = T4_1S

        spatial5 = self.spatial_net.aspp(spatial4)

        temporal5 = self.temporal_net.aspp(temporal4)

        S5_1T, T5_1S = self.fpm5(spatial5, temporal5)

        h5_spatial = S5_1T

        h5_temporal = T5_1S

        h3_spatial_t = self.T_layer3_s(h3_spatial)
        h3_temporal_t = self.T_layer3_t(h3_temporal)

        h4_spatial_t = self.T_layer4_s(h4_spatial)
        h4_temporal_t = self.T_layer4_t(h4_temporal)

        h5_spatial_t = self.T_layer5_s(h5_spatial)
        h5_temporal_t = self.T_layer5_t(h5_temporal)

        h3 = self.net(h3_spatial_t, h3_temporal_t)
        h4 = self.net(h4_spatial_t, h4_temporal_t)
        h5 = self.net(h5_spatial_t, h5_temporal_t)

        h3 = self.squeeze_3(torch.cat((h3, h3_spatial_t, h3_temporal_t), 1))
        h4 = self.squeeze_4(torch.cat((h4, h4_spatial_t, h4_temporal_t), 1))
        h5 = self.squeeze_5(torch.cat((h5, h5_spatial_t, h5_temporal_t), 1))

        return [h3, h4, h5], [h1_spatial, h2_spatial, spatial5], [h1_temporal, h2_temporal, temporal5]

    def decoder(self, fuse_feature, spatial_f, temporal_f):

        feature5 = self.blockaspp(fuse_feature[2])

        B, C, H, W = fuse_feature[1].size()
        feature4 = self.block4(
            torch.cat([fuse_feature[1], F.interpolate(feature5, (H, W), mode='bilinear', align_corners=True)], dim=1))
        B, C, H, W = fuse_feature[0].size()
        feature3 = self.block3(
            torch.cat([fuse_feature[0], F.interpolate(feature4, (H, W), mode='bilinear', align_corners=True)], dim=1))

        B, C, H, W = spatial_f[1].size()
        feature2 = self.block2(torch.cat([self.T_layer2(torch.cat([spatial_f[1], temporal_f[1]], 1)),
                                          F.interpolate(feature3, (H, W), mode='bilinear', align_corners=True)], dim=1))
        B, C, H, W = spatial_f[0].size()
        feature1 = self.block1(torch.cat([self.T_layer1(torch.cat([spatial_f[0], temporal_f[0]], 1)),
                                          F.interpolate(feature2, (H, W), mode='bilinear', align_corners=True)], dim=1))

        B, C, H, W = spatial_f[0].size()

        out5 = F.interpolate(self.pred_head5(feature5), (H * 4, W * 4), mode='bilinear', align_corners=True)
        out4 = F.interpolate(self.pred_head4(feature4), (H * 4, W * 4), mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.pred_head3(feature3), (H * 4, W * 4), mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.pred_head2(feature2), (H * 4, W * 4), mode='bilinear', align_corners=True)
        out1 = F.interpolate(self.pred_head1(feature1), (H * 4, W * 4), mode='bilinear', align_corners=True)

        spred = self.spatial_net.last_conv(spatial_f[-1])

        sout = F.interpolate(spred, (H * 4, W * 4), mode='bilinear', align_corners=True)

        tpred = self.temporal_net.last_conv(temporal_f[-1])

        tout = F.interpolate(tpred, (H * 4, W * 4), mode='bilinear', align_corners=True)

        return out1, out2, out3, out4, out5, sout, tout

    def forward(self, img, flow):
        fuse_feature, spatial_feature, temporal_feature = self.encoder(img, flow)
        return self.decoder(fuse_feature, spatial_feature, temporal_feature)


if __name__ == '__main__':
    img = torch.randn(2, 3, 448, 448).cuda()
    flow = torch.randn(2, 3, 448, 448).cuda()
    net = Interactive().cuda()
    out1 = net(img, flow)
    print(out1[0].shape)
