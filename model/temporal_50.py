import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet50
from .aspp import ASPP


class Temporal(nn.Module):
    def __init__(self, pretrained):
        super(Temporal, self).__init__()
        # -------------Encoder--------------
        resnet = ResNet50(3, 16, pretrained=pretrained)
        # -------------Encoder--------------
        self.inconv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        asppInputChannels = 2048
        asppOutputChannels = 256

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, [1, 6, 12, 18])
        self.last_conv = nn.Sequential(
            nn.Conv2d(asppOutputChannels, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        hx = self.inconv(x)
        h1 = self.encoder1(hx)
        h2 = self.encoder2(h1)
        h3 = self.encoder3(h2)
        h4 = self.encoder4(h3)
        h5 = self.aspp(h4)
        output = self.last_conv(h5)

        return F.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=True)

