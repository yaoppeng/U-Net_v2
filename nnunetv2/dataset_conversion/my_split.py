from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
import numpy as np


def do_split():
    split_file = "/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data/" \
                 "Dataset003_Cirrus/splits_final.json"

    # Dataset003_Cirrus
    # Dataset004_Spectralis
    # Dataset005_Topcon

    data_path = []
    dataset_name = 'Topcon'  # [Spectralis, Cirrus, Topcon]
    path = f"/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/Dataset001_ReTouch/" \
           f"{dataset_name}/TrainingSet-{dataset_name}"
    print("Dataset: {}".format(dataset_name))

    for path in subdirs(path, join=False):
        data_path.append(f"{path}_{dataset_name}")

    split_final = []
    kf = KFold(n_splits=3, shuffle=False)
    for train_path, val_path in kf.split(data_path):
        split_final.append({"train": list(np.array(data_path)[train_path]),
                            "val": list(np.array(data_path)[val_path])})
        # print(f"train_path: {np.array(data_path)[train_path]}")
        # print(f"val_path: {np.array(data_path)[val_path]}")

    save_json(split_final, split_file)


if __name__ == "__main__":
    do_split()
