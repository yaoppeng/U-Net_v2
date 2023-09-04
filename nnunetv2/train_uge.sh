#!/bin/bash
#$ -q gpu@@csecri-titanxp
#$ -o job_logs/$JOB_NAME-$JOB_ID.log
#$ -l gpu_card=1
#$ -pe smp 12

conda activate python39

#export PYTHONPATH=/afs/crc.nd.edu/user/y/ypeng4/MySegCoronary:$PYTHONPATH

# gpu@qa-xp-004
# gpu@qa-p100-002
#gpu@@csecri-titanxp
#python ../../run_training.py --num_gpus 2 --network UNet3d_3d --task 007 \
#        --vols 1

h=$HOSTNAME

#input_dir=/tmp/ypeng4/data/raw_data/Dataset001_ReTouch
#input_dir=/tmp/ypeng4/data/raw_data/Dataset003_Cirrus
#input_dir=/tmp/ypeng4/data/raw_data/Dataset004_Spectralis
#input_dir=/tmp/ypeng4/data/raw_data/Dataset005_Topcon
#input_dir=/tmp/ypeng4/data/preprocessed_data/Dataset002_Iowa

input_dir=/tmp/ypeng4/data/raw_data/Dataset003_Cirrus
output_dir=/tmp/ypeng4/output/Dataset003_Cirrus
mkdir -p $input_dir
mkdir -p $output_dir
#cp -r $HOME/data/raw_data/Dataset001_ReTouch/imagesTr $input_dir
#cp -r $HOME/data/raw_data/Dataset001_ReTouch/labelsTr $input_dir
#cp -r $HOME/data/raw_data/Dataset001_ReTouch/imagesTs $input_dir
cp -r $HOME/data/preprocessed_data/Dataset003_Cirrus /tmp/ypeng4/data/preprocessed_data

export nnUNet_raw=/tmp/ypeng4/data/raw_data
export nnUNet_preprocessed=/tmp/ypeng4/data/preprocessed_data
#export nnUNet_results=/tmp/ypeng4/data/trained_models
#export nnUNet_results=/afs/crc.nd.edu/user/y/ypeng4/data/trained_models
#nnUNetv2_train 001 3d_fullres 4 --npz
nnUNetv2_train 003 2d  0 --npz --c
#nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]

#cp -r $output_dir $HOME/CoronaryClassification
