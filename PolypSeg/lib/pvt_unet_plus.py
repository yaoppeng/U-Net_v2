import time

import numpy as np
from tqdm import tqdm
import future.types
from torch import nn
import torch
# from nnunetv2.training.network.model.dim2.pvt.pvtv2 import *
from lib.pvtv2 import pvt_v2_b2
import timm
import torch.nn.functional as F

__all__ = ['UNet', 'NestedUNet']


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # return self.conv1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # f1, f2, f3, f4 = self.backbone(x)  # (x: 3, 352, 352)


        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        # nb_filter = [32, 64, 128, 256, 512, 512]
        nb_filter = [32, 64, 64, 128, 320, 512]

        self.backbone = pvt_v2_b2()
        path = '/afs/crc.nd.edu/user/y/ypeng4/Polyp-PVT_2/pvt_pth/pvt_v2_b2.pth'

        save_model = torch.load(path)
        print(f"loaded: {path}")
        # self.backbone.load_state_dict(save_model)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        # self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        # self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        # self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        # self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        # self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_1 = VGGBlock(nb_filter[4] + nb_filter[5], nb_filter[4], nb_filter[4])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_2 = VGGBlock(nb_filter[3] * 2 + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_3 = VGGBlock(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_4 = VGGBlock(nb_filter[1] * 4 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_5 = VGGBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.trans_conv1 = nn.Conv2d(64, 64, 1)
        self.trans_conv2 = nn.Conv2d(64, 32, 1)

    def forward(self, input):
        f1, f2, f3, f4 = self.backbone(input)  # (x: 3, 352, 352)

        t1 = self.trans_conv1(F.upsample(f1, scale_factor=2))
        t2 = self.trans_conv2(F.upsample(t1, scale_factor=2))

        x0_0 = t2
        x1_0 = t1
        x2_0 = f1
        x3_0 = f2
        x4_0 = f3
        x5_0 = f4

        # x0_0 = self.conv0_0(input)  # (32, 256, 256)
        # x1_0 = self.conv1_0(self.pool(x0_0))  # (64, 128, 128)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # x2_0 = self.conv2_0(self.pool(x1_0))  # (128, 64, 64)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        # x3_0 = self.conv3_0(self.pool(x2_0))  # (256, 32, 32)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # x4_0 = self.conv4_0(self.pool(x3_0))  # (512, 16, 16)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # x5_0 = self.conv5_0(self.pool(x4_0)) # (512, 8, 8)
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            output5 = self.final5(x0_5)
            return [output1, output2, output3, output4, output5]

        else:
            # output = self.final(x0_4)
            output = self.final(x0_5)
            return output


if __name__ == '__main__':
    model = NestedUNet(num_classes=1)
    x = torch.rand((1, 3, 256, 256))
    model(x)

    from torchinfo import summary

    # summary(model, input_size=(1, 3, 256, 256))

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count
    # flops = FlopCountAnalysis(model, torch.randn(1, 3, 256, 256))
    #
    # print(f"flops: {flops.total()/10**9:.3f}")
    #
    # params = parameter_count_table(model)  # str
    # print(params)

    # print(f"params: {params/10**6:.3f}")

    times = []
    import time

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

