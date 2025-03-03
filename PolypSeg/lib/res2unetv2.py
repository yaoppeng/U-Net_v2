import future.types
from torch import nn
import torch
import timm
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(channel, channel, kernel_size=3,
        stride = 1, padding = 1)] * 4)

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class Res2Network(nn.Module):
    """
    use SpatialAtt + ChannelAtt
    """
    def __init__(self, channel=32, n_classes=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision=deep_supervision

        from lib.Res2Net_v1b import res2net50_v1b_26w_4s
        self.backbone = res2net50_v1b_26w_4s(pretrained=True,
                                             output_stride=16)

        print(f"use SpatialAtt + ChannelAtt and My attention layer".center(80, "="))

        self.ca_1 = ChannelAttention(256)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(512)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(1024)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(2048)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(256, channel, 1)
        self.Translayer_2 = BasicConv2d(512, channel, 1)
        self.Translayer_3 = BasicConv2d(1024, channel, 1)
        self.Translayer_4 = BasicConv2d(2048, channel, 1)

        self.attention_1 = AttentionLayer(channel)
        self.attention_2 = AttentionLayer(channel)
        self.attention_3 = AttentionLayer(channel)
        self.attention_4 = AttentionLayer(channel)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, n_classes, 1, 1)] * 4)

        self.deconv2 = nn.ConvTranspose2d(channel, channel,
                                          kernel_size=3,
                                          stride=1, padding=1,
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel,
                                          kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel,
                                          kernel_size=4, stride=2,
                                          padding=1, bias=False)

        self.deconv5 = nn.ConvTranspose2d(channel, channel,
                                          kernel_size=4, stride=2,
                                          padding=1, bias=False)

    def forward(self, x):
        seg_outs = []

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        f1 = self.backbone.layer1(x)
        f2 = self.backbone.layer2(f1)
        f3 = self.backbone.layer3(f2)
        f4 = self.backbone.layer4(f3)

        # (x: 3, 352, 352)
        # f1: [1, 64, 88, 88]
        # f2: [1, 128, 44, 44]
        # f3: [1, 320, 22, 22]
        # f4: [1, 512, 11, 11]
        f1 = self.ca_1(f1) * f1  # channel attention
        f1 = self.sa_1(f1) * f1  # spatial attention
        f1 = self.Translayer_1(f1)  # (32, 88, 88) 32

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)  # (32, 44, 44) 64

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)  # (32, 22, 22)  128

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)  # (32, 11, 11)  256

        f41 = self.attention_4([f1, f2, f3, f4], f4)

        # f31 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)
        f31 = self.attention_3([f1, f2, f3, f4], f3)

        # f21 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        f21 = self.attention_2([f1, f2, f3, f4], f2)

        # f11 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)
        f11 = self.attention_1([f1, f2, f3, f4], f1)

        y = self.deconv2(f41) + f31  # 44
        y = self.deconv3(y) + f21  # 88

        out1 = self.seg_outs[0](y)  # 88

        y = self.deconv4(y) + f11  # 176
        out2 = self.seg_outs[1](y)  # 176

        return F.interpolate(out1, scale_factor=8, mode='bilinear'), \
            F.interpolate(out2, scale_factor=4, mode='bilinear')


if __name__ == "__main__":
    model = Res2Network(n_classes=2, deep_supervision=True)
    # x = torch.rand((2, 3, 256, 256))  # 448
    # y = model(x)
    # print(len(y))
    # print(y[0].shape, y[1].shape)

    import time
    times = []
    model = model.cuda()
    model.eval()
    import numpy as np
    from tqdm import tqdm

    for i in tqdm(range(100)):
        x1 = torch.randn(1, 3, 256, 256).cuda()

        start = time.time()
        predict = model(x1)
        end = time.time()
        times.append(end - start)

    print(f"FPS: {1.0 / np.mean(times):.3f}")

