## Pytorch implement of UNet_v2

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
python /path/to/UNet_v2/run/run_training.py dataset_id 2d 0 --no-debug -tr ISICTrainer --c
```

### 2. Polyp segmentation

Download the training dataset from [google drive](https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view?usp=sharing) and testing dataset from [google drive](https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing)

run the training and testing using the following command:
```bash
python /path/to/UNet_v2/PolypSeg/Train.py
```
