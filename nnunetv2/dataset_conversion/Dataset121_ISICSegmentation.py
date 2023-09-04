import multiprocessing
import shutil
import numpy as np
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('tkAgg')

    # seg[seg == 255] = 1
    seg[seg > 0] = 1
    # plt.imshow(seg, cmap="gray");
    # plt.show()
    image = io.imread(input_image)
    image = image.sum(2)
    mask = seg
    # mask = image == (3 * 255)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                         sizes[j] > min_component_size])
    mask = binary_fill_holes(mask).astype(np.uint8)
    # seg[mask] = 0

    io.imsave(output_seg, mask, check_contrast=False)
    # io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    # source = '/media/fabian/data/raw_datasets/Massachussetts_road_seg/road_segmentation_ideal'
    # source = "/afs/crc.nd.edu/user/y/ypeng4/EGE-UNet/data/isic2017/"
    # dataset_name = 'Dataset121_ISICSegmentation'

    source = "/afs/crc.nd.edu/user/y/ypeng4/EGE-UNet/data/isic2018/"
    dataset_name = 'Dataset124_ISIC2018'

    orig_img_dir = join(nnUNet_raw, dataset_name, 'all_images')
    orit_gt_dir = join(nnUNet_raw, dataset_name, 'all_gt')

    maybe_mkdir_p(orig_img_dir)
    maybe_mkdir_p(orit_gt_dir)

    # imagestr = join(, 'imagesTr')
    # imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    # labelstr = join(, 'labelsTr')
    # labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    # maybe_mkdir_p(imagestr)
    # maybe_mkdir_p(imagests)
    # maybe_mkdir_p(labelstr)
    # maybe_mkdir_p(labelsts)

    train_source = join(source, 'train')
    test_source = join(source, 'val')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, 'masks'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(train_source, 'images', v.replace("_segmentation.png", ".jpg")),
                         join(train_source, 'masks', v),
                         join(orig_img_dir, "train_" + v.replace("_segmentation.png", ".jpg")[:-4] + '_0000.png'),
                         join(orit_gt_dir, "train_" + v.replace("_segmentation.png", ".png")),
                         50
                     ),)
                )
            )

        # test set
        valid_ids = subfiles(join(test_source, 'masks'), join=False, suffix='png')
        num_train += len(valid_ids)
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(test_source, 'images', v.replace("_segmentation.png", ".jpg")),
                         join(test_source, 'masks', v),
                         join(orig_img_dir, "val_" + v.replace("_segmentation.png", ".jpg")[:-4] + '_0000.png'),
                         join(orit_gt_dir, "val_" + v.replace("_segmentation.png", ".png")),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'disease': 1},
                          num_train, '.jpg', dataset_name=dataset_name)
