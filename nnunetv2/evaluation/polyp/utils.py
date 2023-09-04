import os
import torch
import yaml
import cv2
import argparse

import numpy as np

from easydict import EasyDict as ed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=
    '/afs/crc.nd.edu/user/y/ypeng4/UACANet/configs/UACANet-L.yaml')
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    
    cuda_visible_devices = None
    local_rank = -1

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        cuda_visible_devices = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == -1:
        device_num = 1
    elif cuda_visible_devices is None:
        device_num = torch.cuda.device_count()
    else:
        device_num = len(cuda_visible_devices)

    args.device_num = device_num
    args.local_rank = local_rank

    return args

def load_config(config_dir):
    return ed(yaml.load(open(config_dir), yaml.FullLoader))


def to_cuda(sample):
    for key in sample.keys():
        if type(sample[key]) == torch.Tensor:
            sample[key] = sample[key].cuda()
    return sample


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def debug_tile(out, size=(100, 100)):
    debugs = []
    for debs in out['debug']:
        debug = []
        for deb in debs:
            log = torch.sigmoid(deb).cpu().detach().numpy().squeeze()
            log = (log - log.min()) / (log.max() - log.min())
            log *= 255
            log = log.astype(np.uint8)
            log = cv2.cvtColor(log, cv2.COLOR_GRAY2RGB)
            log = cv2.resize(log, size)
            debug.append(log)
        debugs.append(np.vstack(debug))
    return np.hstack(debugs)

