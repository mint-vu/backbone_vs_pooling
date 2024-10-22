# Source: https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ScanObjectNN/ScanObjectNN.py (Apache License 2.0) (modified from original source)

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        # note that this link only contains the hardest perturbed variant (PB_T50_RS).
        # for full versions, consider the following link.
        www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
        # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget --cipher \'DEFAULT:!DH\' %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_scanobjectnn_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    data_partition = 'training' if partition == 'train' else 'test'
    h5_name = BASE_DIR + '/data/h5_files/main_split/' + data_partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNN(Dataset):
    num_classes = 15

    def __init__(self, num_points, partition='train', dataset_size=0.975, seed=123):
        self.data, self.label = load_scanobjectnn_data(partition)

        self.num_points = num_points
        self.partition = partition

        size = self.data.shape[0]

        if partition in ["train", "valid"]:
            np.random.seed(seed)
            idx = np.arange(size)
            np.random.shuffle(idx)
            train_idx = idx[:int(size * dataset_size)]
            valid_idx = idx[int(size * 0.975):]

        if partition == "train":
            self.data = self.data[train_idx]
            self.label = self.label[train_idx]
        elif partition == "valid":
            self.data = self.data[valid_idx]
            self.label = self.label[valid_idx]

    def __getitem__(self, item):
        pointcloud = self.data[item]
        np.random.shuffle(pointcloud)
        pointcloud = pointcloud[:self.num_points]

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)

        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':

    download()

    train = ScanObjectNN(1024, 'train', 0.01)
    valid = ScanObjectNN(1024, 'valid')
    test = ScanObjectNN(1024, 'test')

    print(len(train))