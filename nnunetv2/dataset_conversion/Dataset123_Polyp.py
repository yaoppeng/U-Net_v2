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
    source = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/polyp"
    dataset_name = 'Dataset123_Polyp'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr_ori')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs_ori')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr_ori')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs_ori')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'TrainDataset')
    test_source = join(source, 'TestDataset')

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
                         join(train_source, 'images', v),
                         join(train_source, 'masks', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )

        # test set
        for sub_data in subdirs(test_source, join=False):
            valid_ids = subfiles(join(test_source, sub_data, 'masks'), join=False, suffix='png')
            num_train += len(valid_ids)
            for v in valid_ids:
                r.append(
                    p.starmap_async(
                        load_and_covnert_case,
                        ((
                             join(test_source, sub_data, 'images', v),
                             join(test_source, sub_data, 'masks', v),
                             join(imagests, sub_data + "_" + v[:-4] + '_0000.png'),
                             join(labelsts, sub_data + "_" + v),
                             50
                         ),)
                    )
                )
            _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, {'background': 0, 'disease': 1},
                          num_train, '.png', dataset_name=dataset_name)
