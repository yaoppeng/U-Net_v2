from typing import List
from torchvision.models.resnet import resnet50, resnet101, resnet34
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from torch import nn
from networks.resunet.decoder import Decoder
import torchvision
import torch

class ResUNet(nn.Module):
    def __init__(self, num_classes, n_conv_per_stage_decoder, deep_supervision, *args, **kwargs):
        super().__init__()
        # self.backbone = resnet50(weights=torchvision.my_models.ResNet50_Weights.IMAGENET1K_V2,
        #                          pretrained=True)

        self.encoder = resnet34(pretrained="pretrained")
        del self.encoder.avgpool, self.encoder.fc
        # self.backbone.layer1.register_forward_hook(self.hook)
        # self.backbone.layer2.register_forward_hook(self.hook)
        # self.backbone.layer3.register_forward_hook(self.hook)
        # self.backbone.layer4.register_forward_hook(self.hook)
        # self.skip = []
        # a = self.encoder.layer4[-1]
        input_features_skip = [
            self.encoder.layer4[-1].bn2.num_features,
            self.encoder.layer3[-1].bn2.num_features,
            self.encoder.layer2[-1].bn2.num_features,
            self.encoder.layer1[-1].bn2.num_features,
            self.encoder.conv1.out_channels
        ]
        self.decoder = Decoder(input_features_skip=input_features_skip)

    def forward(self, x):
        skips = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        out_0 = self.encoder.relu(x)
        skips.append(out_0)

        x = self.encoder.maxpool(out_0)

        out_1 = self.encoder.layer1(x)
        skips.append(out_1)

        out_2 = self.encoder.layer2(out_1)
        skips.append(out_2)

        out_3 = self.encoder.layer3(out_2)
        skips.append(out_3)

        out_4 = self.encoder.layer4(out_3)
        skips.append(out_4)

        output = self.decoder(skips)
        return output[-1]
        # self.backbone(x)
        # return skips

    # def hook(self, module, input, output):
    #     self.skip.append(output)


if __name__ == "__main__":
    x = torch.rand((2, 3, 224, 224)).cuda()
    net = ResUNet(num_classes=4, n_conv_per_stage_decoder=2, deep_supervision=True).cuda()
    y = net(x)

    for item in y:
        print(item.shape)
    # print(len(net(x)))



