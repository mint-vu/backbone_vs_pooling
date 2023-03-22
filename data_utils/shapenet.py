from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'shapenetcore_partanno_segmentation_benchmark_v0')

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenetcore_partanno_segmentation_benchmark_v0')):
        www = 'https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR.replace(' ', '\ ')))
        os.system('rm %s' % (zipfile))


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 npoints=1024,
                 partition='train',
                 data_augmentation=True):
        
        self.npoints = npoints
        self.catfile = os.path.join(ROOT, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.partition = partition
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        split = 'test' if partition == 'test' else 'train'
        splitfile = os.path.join(ROOT, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(ROOT, category, 'points', uuid+'.pts'),
                                        os.path.join(ROOT, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.datapath.sort()

        size = len(self.datapath)
        if partition == "train":
            self.datapath = self.datapath[:int(size * 0.99)]
        elif partition == "valid":
            self.datapath = self.datapath[int(size * 0.99):]

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[fn[0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set, cls

    def __len__(self):
        return len(self.datapath)
