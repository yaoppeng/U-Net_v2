import os

from batchgenerators.utilities.file_and_folder_operations import *

task_dir = "Dataset124_ISIC2018"
# /afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/Dataset124_ISIC2018
# /afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/Dataset124_ISIC2018
split_file = f'/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/' \
             f'{task_dir}/splits_final.json'

os.makedirs(f"/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/{task_dir}", exist_ok=True)
save = True

if save:
    val_dir = "/afs/crc.nd.edu/user/y/ypeng4/EGE-UNet/data/isic2018/val/masks"
    train_dir = "/afs/crc.nd.edu/user/y/ypeng4/EGE-UNet/data/isic2018/train/masks"

    val_cases = subfiles(val_dir, suffix='.png', join=False)
    val_cases = [x.replace("_segmentation.png", "") for x in val_cases]

    train_cases = subfiles(train_dir, suffix='.png', join=False)
    train_cases = [x.replace("_segmentation.png", "") for x in train_cases]

    split = [{'train': train_cases, 'val': val_cases}]
    save_json(split, split_file)
else:
    splits = load_json(split_file)
    print(len(splits))
    print(splits[0].keys())
    print(len(splits[0]['train']))
    print(len(splits[0]['val']))
    print(splits[0])
