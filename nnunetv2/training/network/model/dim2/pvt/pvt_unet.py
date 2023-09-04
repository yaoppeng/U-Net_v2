from torch import nn
import torch.nn.functional as F
import torch

import sys
sys.path.append("/afs/crc.nd.edu/user/y/ypeng4/nnUNet/nnunetv2/training/network/model/dim2")
# import nnunetv2.training.network.model.dim2.segmentation_models_pytorch as smp
import segmentation_models_pytorch as smp


class PVTUNet(nn.Module):
    def __init__(self, n_classes=2, deep_supervision=True):
        super().__init__()
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision

        self.backbone = nn.Identity()
        self.model = smp.Unet(
            encoder_name='pvt',
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes
        )

    def forward(self, x):
        out = self.model(self.backbone(x))

        if self.deep_supervision:
            return [out, F.max_pool2d(out, kernel_size=2, stride=2),
                    F.max_pool2d(out, kernel_size=4, stride=4),
                    F.max_pool2d(out, kernel_size=8, stride=8)]
        else:
            return out


if __name__ == "__main__":
    model = PVTUNet()
    x = torch.rand((2, 3, 256, 256)).cuda()
    y = model(x)
    print(y.shape)
