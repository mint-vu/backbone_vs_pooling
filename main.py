import torch
from torch.multiprocessing import Pool, Process
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


def main(args):

    backbones = args.backbones
    poolings = args.poolings
    num_experiments = args.num_experiments
    experiment_ids = list(1e3 * (1 + np.arange(num_experiments)))
    num_gpus = min(args.num_gpus, torch.cuda.device_count())

    data_utils.download()

    params = []

    i = 0
    for backbone_type, pooling_type, experiment_id in itertools.product(backbones, poolings, experiment_ids):
        # TODO: Configure method to get the backbone and pooling arguments
        backbone_args = {
            "d_in": 3,
            "d_out": 3,
        }
        pooling_args = {}
        params.append((backbone_type, pooling_type, experiment_id, backbone_args, pooling_args, i))
        i = (i + 1) % num_gpus

    print('Total number of experimens:', len(params))

    num_processes = min(10, len(params))

    pool = Pool(num_processes)
    all_results = pool.starmap(train_test, params)
    pool.close()
    pool.join()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--all', action='store_true', help='Run all backbones and poolings', required=False)
    parser.add_argument('-b', '--backbones', nargs="*", type=str, help='List backbone types', required=False)
    parser.add_argument('-p', '--poolings', nargs="*", type=str, help='List pooling types', required=False)
    parser.add_argument('-e', '--num_experiments', type=int, default=1, help='Number of experiments', required=False)
    parser.add_argument('-g', '--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs', required=False)

    args = parser.parse_args()
    validate(args)

    main(args)