#!/bin/bash

#$ -q gpu@@csecri-p100
#$ -o job_logs/$JOB_NAME-$JOB_ID.log
#$ -l gpu_card=1
#$ -pe smp 12

export PYTHONPATH=/afs/crc.nd.edu/user/y/ypeng4/nnUNet:$PYTHONPATH

# gpu@qa-xp-004
# gpu@qa-p100-002
#gpu@@csecri-titanxp
#python ../../run_training.py --num_gpus 2 --network UNet3d_3d --task 007 \
#        --vols 1
export HOME=/afs/crc.nd.edu/user/y/ypeng4

input_raw_dir=/tmp/ypeng4/raw_data
input_pre_dir=/tmp/ypeng4/preprocessed_data
output_dir=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models

mkdir -p $input_raw_dir/Dataset122_ISIC2017
mkdir -p $input_pre_dir/Dataset122_ISIC2017
mkdir -p $output_dir/Dataset122_ISIC2017
#cp -r $HOME/data/raw_data/Task000_TCFA/unfold_from_img_center $input_dir
cp -r $HOME/data/preprocessed_data/Dataset122_ISIC2017 $input_pre_dir
#cp $HOME/data/raw_data/Task000_TCFA/splits_final.pkl $input_dir

export nnUNet_raw=$input_raw_dir
export nnUNet_preprocessed=$input_pre_dir
export nnUNet_results=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models


#    --my_o_dir="$output_dir" \
/afs/crc.nd.edu/user/y/ypeng4/.conda/envs/python38/bin/python \
/afs/crc.nd.edu/user/y/ypeng4/nnUNet/nnunetv2/run/run_training.py 122 2d 0 \
  --no-debug --job_id "$JOB_ID"

#cp -r $output_dir $HOME/CoronaryClassification
