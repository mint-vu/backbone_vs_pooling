import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from data_utils import ModelNet40
from backbones.all_backbones import Backbone
from poolings.all_poolings import Pooling

import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os


batch_size = 32
num_epochs = 200
lr = 1e-3
num_classes = 40


def train_test(backbone_type, pooling_type, num_ref_points, num_projections, num_points_per_set, gpu_index):

    device = f'cuda:{gpu_index}'

    # create results directory if it doesn't exist
    results_dir = 'directory/results/modelnet40/{}/{}_{}_{}/'.format(num_points_per_set, backbone_type, pooling_type, num_ref_points)
    os.makedirs(results_dir, exist_ok=True)

    print("params", backbone_type, pooling_type, num_ref_points,num_projections, num_points_per_set)

    # get the datasets
    phases = ['train', 'test']
    dataset = {}
    for phase in phases:
        dataset[phase] = ModelNet40(num_points_per_set, partition=phase)

    # create the dataloaders
    loader = {}
    for phase in phases:
        if phase == 'train':
            shuffle = True
        else:
            shuffle = False
        loader[phase] = DataLoader(dataset[phase], batch_size=batch_size, shuffle=shuffle)

    # create the modules
    backbone = Backbone(backbone_type=backbone_type, d_in=3)
    pooling = Pooling(pooling=pooling_type, d_in=backbone.d_out, num_projections=num_projections, num_ref_points=num_ref_points)
    classifier = nn.Linear(pooling.num_outputs, num_classes)

    backbone.to(device)
    pooling.to(device)
    classifier.to(device)

    # start training
    criterion = nn.CrossEntropyLoss()

    params =  list(pooling.parameters()) + list(classifier.parameters())
    
    if list(backbone.parameters()):
        params += list(backbone.parameters())
    
    optim = Adam(params, lr=lr)
    scheduler = StepLR(optim, step_size=50, gamma=0.5)

    epochMetrics = defaultdict(list)
    for epoch in tqdm(range(num_epochs)):

        for phase in phases:

            if phase == 'train':
                backbone.train()
                pooling.train()
                classifier.train()
            else:
                backbone.eval()
                pooling.eval()
                classifier.eval()

            loss_ = []
            acc_ = []

            for i, data in enumerate(loader[phase]):

                # zero the parameter gradients
                optim.zero_grad()

                x, y = data

                x = x.to(device).to(torch.float)
                y = y.to(device).squeeze()
                #print(x.shape)
                #print(y.shape)

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # pass the sets through the backbone and pooling
                    z = backbone(x)
                    v = pooling(z)
                    logits = classifier(v)
                    loss = criterion(logits, y)

                    acc = (1. * (torch.argmax(logits, dim=1) == y)).mean().item()
                    #print(acc)

                    # backpropogation only in training phase
                    if phase == 'train':
                        # Backward pass
                        print(True)
                        loss.backward(retain_graph=True)
                        print(True)
                        # 1-step gradient descent
                        optim.step()

                # save losses and accuracies
                loss_.append(loss.item())
                acc_.append(acc)
                

            epochMetrics[phase, 'loss'].append(np.mean(loss_))
            epochMetrics[phase, 'acc'].append(np.mean(acc_))

        scheduler.step()
        
        print(epochMetrics)
        # save intermediate results so far
        torch.save(epochMetrics, results_dir + '{}.json'.format(num_ref_points))

    return epochMetrics