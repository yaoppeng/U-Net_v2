import os

from batchgenerators.utilities.file_and_folder_operations import *

split_file = '/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/Dataset123_Polyp/splits_final.json'
os.makedirs("/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/Dataset123_Polyp", exist_ok=True)
save = True

if save:
    val_dir = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/Dataset123_Polyp/labelsTs_ori"
    train_dir = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/Dataset123_Polyp/labelsTr_ori"

    val_cases = subfiles(val_dir, suffix='.png', join=False)
    val_cases = [x.replace(".png", "") for x in val_cases]

    train_cases = subfiles(train_dir, suffix='.png', join=False)
    train_cases = [x.replace(".png", "") for x in train_cases]

    split = [{'train': train_cases, 'val': val_cases}]
    save_json(split, split_file)
else:
    splits = load_json(split_file)
    print(len(splits))
    print(splits[0].keys())
    print(len(splits[0]['train']))
    print(len(splits[0]['val']))
    print(splits[0])
