# Source: https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/data.py (MIT License)

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR.replace(' ', '\ ')))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def rotate(pointcloud):
    theta = np.pi * (np.random.uniform() - 0.5) / 3 # between -30deg and 30deg
    rot = np.zeros((3, 3))
    rot[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot[2, 2] = 1
    pointcloud = np.matmul(pointcloud, rot)
    return pointcloud

class ModelNet40(Dataset):
    num_classes = 40
    classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    def __init__(self, num_points, partition='train', dataset_size=0.975, seed=123):
        self.data, self.label = load_data('test' if partition=='test' else 'train')

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
            pointcloud = rotate(pointcloud)

        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':

    download()

    train = ModelNet40(1024, 'train', 0.05)

    print(len(train.label))