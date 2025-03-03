from joblib import Parallel, delayed

import cv2
from skimage import io
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *

dataset_name = "Dataset124_ISIC2018"
root_dir = f"/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/{dataset_name}"
orig_img_dir = join(root_dir, 'all_images')
orit_gt_dir = join(root_dir, 'all_gt')

new_img_dir = join(root_dir, 'all_images_resized')
new_gt_dir = join(root_dir, 'all_gt_resized')


img_files = subfiles(orig_img_dir)
gt_files = subfiles(orit_gt_dir)


def resize_img_or_gt(file_name, mode):
    img = io.imread(file_name)
    assert mode in ['img', 'gt']

    resize_mode = cv2.INTER_AREA if mode == "img" else cv2.INTER_NEAREST
    save_file = file_name.replace("all_images", "all_images_resized") if mode == 'img' else \
        file_name.replace("all_gt", "all_gt_resized")

    img = cv2.resize(img, dsize=(256, 256), interpolation=resize_mode)
    io.imsave(save_file, img)

os.makedirs(new_img_dir, exist_ok=True)
Parallel(-1)(delayed(resize_img_or_gt)(file, 'img') for file in tqdm(img_files))

os.makedirs(new_gt_dir, exist_ok=True)
Parallel(-1)(delayed(resize_img_or_gt)(file, 'gt') for file in tqdm(gt_files))
