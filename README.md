## Pytorch implementation of U-Net v2: RETHINKING THE SKIP CONNECTIONS OF U-NET FOR MEDICAL IMAGE SEGMENTATION

<<<<<<< HEAD
## Pytorch implement of UNet_v2
=======
<<<<<<< HEAD
## Pytorch implement of UNet_v2: RETHINKING THE SKIP CONNECTION OF U-NET FOR MEDICAL IMAGE SEGMENTATION
=======
<<<<<<< HEAD
## Pytorch implement of U-Net_v2: RETHINKING THE SKIP CONNECTION OF U-NET FOR MEDICAL IMAGE SEGMENTATION
=======
#### nnUNet is the GOAT! Thanks to Fabian et al. for making pure U-Net great again. Less is more.
>>>>>>> 858b41d (Update README.md)
>>>>>>> 6a851f9 ( Update README.md)
>>>>>>> 37e9920 (  Update README.md)

<<<<<<< HEAD
=======
Please make sure you have installed all the packages with the correct versions as shown in `requirements.txt`. Most of the issues are caused by incompatible package versions.

>>>>>>> cdfc9bb (Update README.md)
### 1 ISIC segmentation

<<<<<<< HEAD
Down the dataset from ![google drive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)
=======
Download the dataset from [google drive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)
>>>>>>> 6ae9fd8 (Update README.md)

set the nnUNet_raw, nnUNet_preprocessed and nnUNet_results environment variable using the following command:

```bash
export nnUNet_raw=/path/to/input_raw_dir
export nnUNet_preprocessed=/path/to/preprocessed_dir
export nnUNet_results=/path/to/result_save_dir
```

run the training and testing using the following command:
```bash
python /path/to/UNet_v2/run/run_training.py dataset_id 2d 0 --no-debug -tr ISICTrainer --c
```

### 2. Polyp segmentation

Download the training dataset from ![google drive](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing) and testing dataset from ![google drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing)

run the training and testing using the following command:
```bash
python /path/to/UNet_v2/PolypSeg/Train.py
```
<<<<<<< HEAD
=======

<<<<<<< HEAD
=======
### 4. On your own data

I only used the 4× downsampled results on my dataset. You may need to modify the code:

```
f1, f2, f3, f4, f5, f6 = self.encoder(x)

...
f61 = self.sdi_6([f1, f2, f3, f4, f5, f6], f6)
f51 = self.sdi_5([f1, f2, f3, f4, f5, f6], f5)
f41 = self.sdi_4([f1, f2, f3, f4, f5, f6], f4)
f31 = self.sdi_3([f1, f2, f3, f4, f5, f6], f3)
f21 = self.sdi_2([f1, f2, f3, f4, f5, f6], f2)
f11 = self.sdi_1([f1, f2, f3, f4, f5, f6], f1)
```

and delete the following code:

```
for i, o in enumerate(seg_outs):
    seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')
```

By doing this, you are using all the resolution results rather than the 4× downsampled ones.

The following code snippet shows how to use `U-Net v2` in training and testing.

For training:

```python
from unet_v2.UNet_v2 import *

n_classes=2
pretrained_path="/path/to/pretrained/pvt"
model = UNetV2(n_classes=, deep_supervision=True, pretrained_path=pretrained_path)

x = torch.rand((2, 3, 256, 256))

ys = model(x)  # ys is a list because of deep supervision

```

Next you can use `ys` and `label` to compute the loss and do back-propagation.

In the testing phase:

```python
model.eval()
model.deep_supervision = False

x = torch.rand((2, 3, 256, 256))
y = model(x)  # y is a tensor since the deep supervision is turned off in the testing phase
print(y.shape)  # (2, n_classes, 256, 256)

pred = torch.argmax(y, dim=1)
```

>>>>>>> 00f57af (Update README.md)
for convience, the `U-Net v2` model file is copied to `./lib/UNet_v2.py`
>>>>>>> 816bad3 ( Update README.md)
