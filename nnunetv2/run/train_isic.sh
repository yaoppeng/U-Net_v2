#!/bin/bash

#export nnUNet_raw=/afs/crc.nd.edu/user/y/ypeng4/data/raw_data
#export nnUNet_preprocessed=/afs/crc.nd.edu/user/y/ypeng4/data/preprocessed_data
#export nnUNet_results=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models

export HOME=/afs/crc.nd.edu/user/y/ypeng4
export PYTHONPATH=/afs/crc.nd.edu/user/y/ypeng4/nnUNet:$PYTHONPATH

input_raw_dir=/tmp/ypeng4/raw_data
input_pre_dir=/tmp/ypeng4/preprocessed_data

output_dir=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models
output_dir_tmp=/tmp/ypeng4/data/trained_models

mkdir -p $input_raw_dir/Dataset122_ISIC2017
mkdir -p $input_pre_dir/Dataset122_ISIC2017
mkdir -p $output_dir_tmp/Dataset122_ISIC2017

cp -r $HOME/data/preprocessed_data/Dataset122_ISIC2017 $input_pre_dir
cp -r $HOME/data/raw_data/polyp $input_raw_dir

export nnUNet_raw=$input_raw_dir
export nnUNet_preprocessed=$input_pre_dir

#export nnUNet_results=$output_dir_tmp
export nnUNet_results=$output_dir

/scratch365/ypeng4/software/bin/anaconda/envs/python310/bin/python \
/afs/crc.nd.edu/user/y/ypeng4/nnUNet/nnunetv2/run/run_training.py 122 2d 0 \
  --no-debug -tr ISICTrainer --job_id "$1" --c

#cp -r $output_dir_tmp/Dataset122_ISIC2017 $output_dir
#rm -rf $output_dir_tmp/Dataset122_ISIC2017
