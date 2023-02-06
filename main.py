from torch.multiprocessing import Pool, Process
import numpy as np
import itertools

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'backbones'))
sys.path.append(os.path.join(BASE_DIR, 'poolings'))

import data_utils

from train import train_test

def main():

    backbones = ['IDT']
    poolings = ['MAX']
    experiment_ids = list(1e3 * (1 + np.arange(1))) # 10 random seeds


    data_utils.download()

    params = []
    for backbone_type, pooling_type, experiment_id in itertools.product(backbones, poolings, experiment_ids):
        # TODO: Configure method to get the backbone and pooling arguments
        backbone_args = {
            "d_in": 3,
            "d_out": 3,
        }
        pooling_args = {}
        params.append((backbone_type, pooling_type, experiment_id, backbone_args, pooling_args, 0))

    print('Total number of experimens:', len(params))

    num_processes = min(10, len(params))

    pool = Pool(num_processes)
    all_results = pool.starmap(train_test, params)
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()