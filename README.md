### nnUNet is the GOAT! Thanks to Fabian et al. for making pure U-Net great again. Less is more.

## Pytorch implement of U-Net v2: RETHINKING THE SKIP CONNECTION OF U-NET FOR MEDICAL IMAGE SEGMENTATION

Please make sure you have installed all the packages with the correct versions as shown in requirements.txt. Most of the issues are caused by incompatible package versions.

The pretrained PVT model: [google drive](https://drive.google.com/drive/folders/1xC5Opwu5Afz4xiK5O9v4NnQOZY0A9-2j?usp=sharing)
### 1 ISIC segmentation

Down the dataset from [google drive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)

set the nnUNet_raw, nnUNet_preprocessed and nnUNet_results environment variable using the following command:

```bash
export nnUNet_raw=/path/to/input_raw_dir
export nnUNet_preprocessed=/path/to/preprocessed_dir
export nnUNet_results=/path/to/result_save_dir
```

run the training and testing using the following command:
```bash
python /path/to/U-Net_v2/run/run_training.py dataset_id 2d 0 --no-debug -tr ISICTrainer --c
```

The `nnUNet` preprocessed data can be downloaded from [ISIC 2017](https://drive.google.com/drive/folders/1Q8VQXhQd5T4Z7kS2SnqygedtYSJSSN75?usp=sharing) and [ISIC 2018](https://drive.google.com/drive/folders/1LMJsdvGDEYRJbX3XQAcjYuOIYhlhvtQF?usp=drive_link)

### 2. Polyp segmentation

Download the training dataset from [google drive](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing) and testing dataset from [google drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing)

run the training and testing using the following command:
```bash
python /path/to/U-Net_v2/PolypSeg/Train.py
```

### 3. On your own data

The following code snippet shows how to use `U-Net v2` in training and testing.

For training:

```python
from unet_v2.UNet_v2 import *

n_classes=2
pretrained_path="/path/to/pretrained/pvt"
model = UNetV2(n_classes=n_classes, deep_supervision=True, pretrained_path=pretrained_path)

x = torch.rand((2, 3, 256, 256))

ys = model(x)  # ys is a list because of deep supervision

```

Now you can use `ys` and `label` to compute the loss and do back-propagation.

In the testing phase:

```python
model.eval()
model.deep_supervision = False

x = torch.rand((2, 3, 256, 256))
y = model(x)  # y is a tensor since the deep supervision is turned off in the testing phase
print(y.shape)  # (2, n_classes, 256, 256)

pred = torch.argmax(y, dim=1)
```

for convience, the `U-Net v2` model file has been copied to `./unet_v2/UNet_v2.py`
