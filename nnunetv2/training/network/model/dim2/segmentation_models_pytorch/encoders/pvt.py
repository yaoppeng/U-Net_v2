from copy import deepcopy

import torch.nn as nn
import torch
from torchvision.models.resnet import ResNet
from segmentation_models_pytorch.encoders.pvtv2 import *
import torch.nn.functional as F
from ._base import EncoderMixin


class PVTEncoder(EncoderMixin, nn.Module):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self.backbone = pvt_v2_b2()

        self.backbone = pvt_v2_b2()
        path = "/afs/crc.nd.edu/user/y/ypeng4/Polyp-PVT_2/pvt_pth/pvt_v2_b2.pth"
        print(f"loading {path}")
        save_model = torch.load(path)
        # self.backbone.load_state_dict(save_model)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

    def forward(self, x):
        f1, f2, f3, f4 = self.backbone(x)  # (x: 3, 352, 352)
        return [x, F.interpolate(f1, scale_factor=2), f1, f2, f3, f4]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


pvt_encoders = {
    "pvt": {
        "encoder": PVTEncoder,
        "params": {
            "out_channels": (3, 64, 64, 128, 320, 512),
        },
    },

}
