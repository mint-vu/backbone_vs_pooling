import torch
from torch.multiprocessing import Pool
import numpy as np
import itertools
import argparse

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'backbones'))
sys.path.append(os.path.join(BASE_DIR, 'poolings'))

import data_utils

from train import train_test
from backbones.all_backbones import BACKBONES
from poolings.all_poolings import POOLINGS

def validate(args):
    if args.all:
        args.backbones = BACKBONES
        args.poolings = POOLINGS
        return
    else:
        if not args.backbones or not args.poolings:
            raise ValueError('Backbones and poolings must be specified')

    backbones = args.backbones
    poolings = args.poolings

    for idx, backbone in enumerate(backbones):
        backbones[idx] = backbone.lower()
        if backbones[idx] not in BACKBONES:
            raise ValueError('Backbone not supported: ' + backbone)

    for idx, pooling in enumerate(poolings):
        poolings[idx] = pooling.lower()
        if poolings[idx] not in POOLINGS:
            raise ValueError('Pooling not supported: ' + pooling)
        
    args.optimizer = args.optimizer.lower()

    if args.optimizer not in ['adam', 'sgd']:
        raise ValueError(f'Optimizer not supported: {args.optimizer}. Supported optimizers: adam, sgd.')

    for gpu in args.gpus:
        if not 0 <= gpu < torch.cuda.device_count():
            raise ValueError('GPU index out of range: ' + str(gpu))

def main(args):

    backbones = args.backbones
    poolings = args.poolings
    num_experiments = args.num_experiments
    experiment_ids = list(1e3 * (1 + np.arange(num_experiments)))
    gpus = args.gpus

    data_utils.download()

    params = []

    gpu_idx = 0
    for backbone_type, pooling_type, experiment_id in itertools.product(backbones, poolings, experiment_ids):
        # NOTE: Using default configurations for testing, this should probably be changed later
        backbone_args = {
            "d_in": 3,
            "d_out": 3,
        }
        pooling_args = {}

        params.append((backbone_type, pooling_type, experiment_id, args.optimizer, backbone_args, pooling_args, gpus[gpu_idx]))
        gpu_idx = (gpu_idx + 1) % len(gpus)

    print('Total number of experiments:', len(params))

    num_processes = min(10, len(params))

    pool = Pool(num_processes)
    all_results = pool.starmap(train_test, params)
    pool.close()
    pool.join()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--all', action='store_true', help='Run all backbones and poolings', required=False)
    parser.add_argument('-b', '--backbones', nargs="*", type=str, default=['idt'], help='List backbone types', required=False)
    parser.add_argument('-p', '--poolings', nargs="*", type=str, default=['max'], help='List pooling types', required=False)
    parser.add_argument('-e', '--num_experiments', type=int, default=1, help='Number of experiments', required=False)
    parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer (either adam or sgd)', required=False)
    parser.add_argument('-g', '--gpus', type=int, nargs="*", default=list(range(torch.cuda.device_count())), help='GPUs to use', required=False)

    args = parser.parse_args()
    validate(args)

    main(args)