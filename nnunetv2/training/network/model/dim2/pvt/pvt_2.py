import future.types
from torch import nn
import torch
# from nnunetv2.training.network.model.dim2.pvt.pvtv2 import *
from nnunetv2.training.network.model.dim2.pvt.pvtv2 import pvt_v2_b2
import timm
import torch.nn.functional as F
from nnunetv2.training.network.model.dim2.pvt.gcn_fuse import _GlobalConvModule
# from timm.my_models.vision_transformer import vit_small_patch16_224, VisionTransformer
# model = vit_small_patch16_224()
#
# x = torch.rand((2, 3, 224, 224))
# y = model(x)
# print(y.shape)


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
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(32, 32, kernel_size=3,
        stride = 1, padding = 1) for _ in range(4)])

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


class PVTNetwork_2(nn.Module):
    """
    use Conv2d(7, 1) Conv2d(1, 7) to replace SpatialConv+ChannelConv
    """
    def __init__(self, channel=32, n_classes=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        print(f"use Conv2d(7, 1) Conv2d(1, 7) and My attention layer".center(80, "="))
        self.backbone = pvt_v2_b2()
        path = '/afs/crc.nd.edu/user/y/ypeng4/Polyp-PVT_2/pvt_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        # self.backbone.load_state_dict(save_model)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer_1 = _GlobalConvModule(64, channel, (7, 7))
        self.Translayer_2 = _GlobalConvModule(128, channel, (7, 7))
        self.Translayer_3 = _GlobalConvModule(320, channel, (7, 7))
        self.Translayer_4 = _GlobalConvModule(512, channel, (7, 7))

        self.attention_1 = AttentionLayer()
        self.attention_2 = AttentionLayer()
        self.attention_3 = AttentionLayer()
        self.attention_4 = AttentionLayer()

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(channel, n_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2, padding=1,
                                          bias=False)
        self.deconv3 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(channel, channel, kernel_size=4, stride=2,
                                          padding=1, bias=False)

    def forward(self, x):
        seg_outs = []
        f1, f2, f3, f4 = self.backbone(x)  # (x: 3, 352, 352)
        # f1: [1, 64, 88, 88]
        # f2: [1, 128, 44, 44]
        # f3: [1, 320, 22, 22]
        # f4: [1, 512, 11, 11]

        f1 = self.Translayer_1(f1)  # (32, 88, 88) 32

        f2 = self.Translayer_2(f2)  # (32, 44, 44) 64

        f3 = self.Translayer_3(f3)  # (32, 22, 22)  128

        f4 = self.Translayer_4(f4)  # (32, 11, 11)  256

        f41 = self.attention_4([f1, f2, f3, f4], f4)
        seg_outs.append(self.seg_outs[0](f41))

        # f31 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)
        f31 = self.attention_3([f1, f2, f3, f4], f3)

        # f21 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        f21 = self.attention_2([f1, f2, f3, f4], f2)

        # f11 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)
        f11 = self.attention_1([f1, f2, f3, f4], f1)

        y = self.deconv2(f41) + f31  # 44

        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv3(y) + f21  # 88

        seg_outs.append(self.seg_outs[2](y))

        y = self.deconv4(y) + f11  # 176

        seg_outs.append(self.seg_outs[3](y))

        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        if self.deep_supervision:
            return seg_outs[::-1]
        else:
            return seg_outs[-1]


if __name__ == "__main__":
    model = PVTNetwork_2(n_classes=1)
    x = torch.rand((2, 3, 256, 256))  # 448
    y = model(x)
    print(y[0].shape, y[1].shape)
