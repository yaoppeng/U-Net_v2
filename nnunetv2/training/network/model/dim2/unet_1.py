import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.network.model.dim2.utils import get_block, get_norm
from nnunetv2.training.network.model.dim2.unet_utils import inconv, down_block, up_block


def trans_conv(all_list, shape):
    # all = [x2, x3, x4, x5, x6, x7]
    ans = []
    for x in all_list:
        if x.shape[-1] < shape:
            ans.append(F.interpolate(x, (shape, shape)))
        else:
            ans.append(F.adaptive_max_pool2d(x, (shape, shape)))
    ans = torch.cat(ans, dim=1)

    return ans

class UNet(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=16,
                 block='SingleConv', pool=True, deep_supervision=True, use_my_net=False):
        super().__init__()
        self.use_my_net = use_my_net

        self.deep_supervision = deep_supervision

        block = get_block(block)
        nb = 2  # num_block

        self.inc = inconv(in_ch, base_ch, block=block)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=nb, block=block,  pool=pool)

        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=nb, block=block, pool=pool)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=nb, block=block, pool=pool)
        self.down4 = down_block(8*base_ch, 16*base_ch, num_block=nb, block=block, pool=pool)
        self.down5 = down_block(16 * base_ch, 32 * base_ch, num_block=nb, block=block, pool=pool)

        self.down6 = down_block(32*base_ch, 32*base_ch, num_block=nb, block=block, pool=pool)

        self.up0_0 = up_block(32 * base_ch, 32*base_ch, num_block=nb, block=block)

        if self.use_my_net:
            self.trans_conv_1 = nn.Conv2d(63*base_ch, 32*base_ch, 1)
            self.trans_conv_2 = nn.Conv2d(63 * base_ch, 16 * base_ch, 1)
            self.trans_conv_3 = nn.Conv2d(63 * base_ch, 8 * base_ch, 1)
            self.trans_conv_4 = nn.Conv2d(63 * base_ch, 4 * base_ch, 1)
            self.trans_conv_5 = nn.Conv2d(63 * base_ch, 2 * base_ch, 1)
            self.trans_conv_6 = nn.Conv2d(63 * base_ch, 1 * base_ch, 1)

        self.up0 = up_block(32*base_ch, 16*base_ch, num_block=nb, block=block)
        self.up1 = up_block(16*base_ch, 8*base_ch, num_block=nb, block=block)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=nb, block=block)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=nb, block=block)
        self.up4 = up_block(2*base_ch, base_ch, num_block=nb, block=block)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

        seg_layers = []
        for i in [32, 16, 8, 4, 2, 1]:
            seg_layers.append(nn.Conv2d(base_ch*i,
                                        num_classes, 1, 1, 0,
                                        bias=True))

        self.seg_layers = nn.ModuleList(seg_layers)
    def forward(self, x): 
        x1 = self.inc(x)  # (H, W)
        x2 = self.down1(x1)  # (H/2, W/2)
        x3 = self.down2(x2)  # (H/4, W/4)
        x4 = self.down3(x3)  # (H/8, W/8)
        x5 = self.down4(x4)  # (H/16, W/16)
        x6 = self.down5(x5)  # (H/32, W/32)

        x7 = self.down6(x6)  # (H/64, W/64)
        # print(f"x6.shape: {x6.shape}")
        seg_out = []

        if self.use_my_net:
            all = [x1, x2, x3, x4, x5, x6]

        if self.use_my_net:
            out = self.up0_0(x7, self.trans_conv_1(trans_conv(all, x6.shape[-1])))
        else:
            # old
            out = self.up0_0(x7, x6)
        seg_out.append(self.seg_layers[0](out))

        if self.use_my_net:
            out = self.up0(out, self.trans_conv_2(trans_conv(all, x5.shape[-1])))
        else:
            # old (x6, x5)
            out = self.up0(out, x5)  # 1st out, (H/16, W/16)
        seg_out.append(self.seg_layers[1](out))

        if self.use_my_net:
            out = self.up1(out, self.trans_conv_3(trans_conv(all, x4.shape[-1])))
        else:
            # old
            out = self.up1(out, x4)  # 2nd out, (H/8, W/8)
        seg_out.append(self.seg_layers[2](out))

        if self.use_my_net:
            out = self.up2(out, self.trans_conv_4(trans_conv(all, x3.shape[-1])))
        else:
            # old
            out = self.up2(out, x3)  # 3rd out,  (H/4, W/4)
        seg_out.append(self.seg_layers[3](out))

        # a = trans_conv(all, x2.shape[-1])
        if self.use_my_net:
            out = self.up3(out, self.trans_conv_5(trans_conv(all, x2.shape[-1])))
        else:
            # old
            out = self.up3(out, x2)  # 4th out, (H/2, W/ 2)
        seg_out.append(self.seg_layers[4](out))

        if self.use_my_net:
            out = self.up4(out, self.trans_conv_6(trans_conv(all, x1.shape[-1])))
        else:
            # old
            out = self.up4(out, x1)  # 5-th out  (C, H, W)
        # seg_out.append(self.seg_layers[4](out))

        out = self.outc(out)  # (cls, H, W)
        seg_out.append(out)

        if self.deep_supervision:
            return seg_out[::-1]
        else:
            return seg_out[-1]


if __name__ == "__main__":
    x = torch.rand((2, 3, 256, 256))  # .cuda()
    # ConvNeXtBlock
    net = UNet(in_ch=3, num_classes=2, block="FusedMBConv")   # .cuda()
    y = net(x)
    print(y[0].shape)

