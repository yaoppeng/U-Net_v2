import os

from joblib import Parallel, delayed

import cv2
from skimage import io
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *

root_dir = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/Dataset123_Polyp/"

new_img_dir = join(root_dir, 'imagesTr')
new_gt_dir = join(root_dir, 'labelsTr')

img_files = subfiles(join(root_dir, 'imagesTr_ori')) + subfiles(join(root_dir, 'imagesTs_ori'))
gt_files = subfiles(join(root_dir, 'labelsTr_ori')) + subfiles(join(root_dir, 'labelsTs_ori'))


def resize_img_or_gt(file_name, mode):
    img = io.imread(file_name)
    assert mode in ['img', 'gt']

    resize_mode = cv2.INTER_AREA if mode == "img" else cv2.INTER_NEAREST
    save_file = join(new_img_dir, os.path.basename(file_name)) if mode == 'img' else \
        join(new_gt_dir, os.path.basename(file_name))

    img = cv2.resize(img, dsize=(352, 352), interpolation=resize_mode)
    io.imsave(save_file, img)

os.makedirs(new_img_dir, exist_ok=True)
Parallel(-1)(delayed(resize_img_or_gt)(file, 'img') for file in tqdm(img_files))

os.makedirs(new_gt_dir, exist_ok=True)
Parallel(-1)(delayed(resize_img_or_gt)(file, 'gt') for file in tqdm(gt_files))
