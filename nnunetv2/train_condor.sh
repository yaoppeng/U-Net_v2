#!/bin/bash

export HOME=/afs/crc.nd.edu/user/y/ypeng4

#/scratch365/ypeng4/software/bin/anaconda/bin/conda init bash

/scratch365/ypeng4/software/bin/anaconda/envs/python310/bin/python \
/afs/crc.nd.edu/user/y/ypeng4/nnUNet/nnunetv2/run/run_training.py 122 2d 0 --no-debug\
  --cluster_id $1
