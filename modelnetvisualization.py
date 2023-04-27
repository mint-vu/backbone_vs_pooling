import os
import numpy as np
from data_utils.modelnet import ModelNet40, download
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_pointcloud(points, save_path='imgs/chair_pointcloud.png'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', s=5, c='b', alpha=0.5)
    
    max_range = np.max(np.abs(points))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)


    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    ax.zaxis.line.set_visible(False)

    fig.patch.set_facecolor('white')
    
    ax.view_init(elev=20, azim=45)
    
    plt.savefig(save_path)


def get_chair_pointclouds(modelnet40_dataset, num_chairs=5):
    chair_pointclouds = []
    chair_count = 0
    for i in range(len(modelnet40_dataset)):
        data, label = modelnet40_dataset[i]
        label = int(label)
        if modelnet40_dataset.classes[label] == 'chair':
            chair_pointclouds.append(data)
            chair_count += 1
            if chair_count >= num_chairs:
                break
    if chair_count == 0:
        raise ValueError("No chair found in the dataset")
    return chair_pointclouds

def rotate_pointcloud_x(points, angle):
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(angle_rad), np.sin(angle_rad)],
                                [0, -np.sin(angle_rad), np.cos(angle_rad)]])
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points

if __name__ == "__main__":
    modelnet40_dataset = ModelNet40(num_points=2048)
    chair_pointclouds = get_chair_pointclouds(modelnet40_dataset)
    
    for i, chair_pointcloud in enumerate(chair_pointclouds):
        rotated_chair_pointcloud = rotate_pointcloud_x(chair_pointcloud, 90)
        save_path = f'imgs/chair_pointcloud_{i + 1}.png'
        visualize_pointcloud(rotated_chair_pointcloud, save_path)

