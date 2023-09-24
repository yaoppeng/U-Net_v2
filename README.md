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
