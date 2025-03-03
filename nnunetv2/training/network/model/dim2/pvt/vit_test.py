from timm.models.vision_transformer import vit_small_patch8_224, VisionTransformer
import torch


model = vit_small_patch8_224()

x = torch.rand((2, 3, 224, 224))
y = model(x)
print(y.shape)
