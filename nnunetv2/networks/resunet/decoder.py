import torch
from torch import nn
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


class Decoder(nn.Module):
    def __init__(self, input_features_skip, num_classes=4):
        super().__init__()
        self.convs = [
            StackedConvBlocks(
                num_convs=2, conv_op=nn.Conv2d,
                input_channels=input_features_skip[i],
                output_channels=input_features_skip[i]//2,
                kernel_size=3,
                initial_stride=1,
                conv_bias=False,
                norm_op=nn.InstanceNorm2d,
                norm_op_kwargs={"eps": 1e-05, "momentum": 0.1,
                                "affine": True, "track_running_stats": False},
                dropout_op=None,
                dropout_op_kwargs={}, nonlin=nn.LeakyReLU,
                nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
                nonlin_first=False
            ) for i in range(len(input_features_skip))
        ]

        self.convs = nn.ModuleList(self.convs)

        self.local_conv = nn.ModuleList([
            nn.Conv2d(input_features_skip[i],
                      input_features_skip[i]//2,
                      stride=1,
                      kernel_size=1,
                      bias=False)
            for i in range(len(input_features_skip))
        ])

        self.seg_layers = nn.ModuleList([
            nn.Conv2d(input_features_skip[i]//2, num_classes,
                      kernel_size=(1, 1), stride=(1, 1))
            for i in range(len(input_features_skip))])

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for i in range(len(skips)-1):
            x = torch.nn.functional.upsample(lres_input, scale_factor=2)
            x = self.local_conv[i](x)
            x = torch.cat([x, skips[-(i+2)]], dim=1)

            lres_input = self.convs[i](x)
            seg_outputs.append(self.seg_layers[i](lres_input))
        return seg_outputs
