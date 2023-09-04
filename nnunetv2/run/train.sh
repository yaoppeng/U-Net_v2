#!/bin/bash

export nnUNet_raw=/afs/crc.nd.edu/user/y/ypeng4/data/raw_data
export nnUNet_preprocessed=/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data
export nnUNet_results=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models
export HOME=/afs/crc.nd.edu/user/y/ypeng4

/scratch365/ypeng4/software/bin/anaconda/envs/python310/bin/python \
/afs/crc.nd.edu/user/y/ypeng4/nnUNet/nnunetv2/run/run_training.py 122 2d 0 \
  --no-debug --job_id "$1"
