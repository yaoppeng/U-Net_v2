#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *


def export_for_submission(source_dir, target_dir):
    """
    promise wants mhd :-/
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    target_files = [join(target_dir, i[:-7] + ".mhd") for i in files]
    maybe_mkdir_p(target_dir)
    for f, t in zip(files, target_files):
        img = sitk.ReadImage(join(source_dir, f))
        sitk.WriteImage(img, t)


if __name__ == "__main__":
    """
    Dataset003_Cirrus
    Dataset004_Spectralis
    Dataset005_Topcon
    """
    folder = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/Dataset005_Topcon"
    out_folder = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/Dataset005_Topcon"

    json_dict = OrderedDict()
    json_dict['name'] = "ReTouch"
    json_dict['description'] = "RetinalFluid"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "OCT",
    }
    json_dict['labels'] = {
        "background": 0,
        "IRF": 1,
        "SRF": 2,
        "PED": 3
    }

    json_dict['channel_names'] = {
        0: "OCT"
    }
    # json_dict['channel_names'] = {
    #                     "0": "OCT"
    #                 },
    json_dict['file_ending'] = ".nii.gz"

    raw_datas = subfiles(join(out_folder, "imagesTr"), suffix='nii.gz')
    test_datas = subfiles(join(out_folder, "imagesTs"), suffix='nii.gz')
    json_dict['numTraining'] = len(raw_datas)
    json_dict['numTest'] = len(test_datas)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-len("_0000.nii.gz")],
                              "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-len("_0000.nii.gz")]}
                             for i in raw_datas]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-len("_0000.nii.gz")]
                         for i in test_datas]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))
