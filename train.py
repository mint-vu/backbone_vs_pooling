import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'backbones'))
sys.path.append(os.path.join(BASE_DIR, 'poolings'))

from data_utils.modelnet import ModelNet40
from data_utils.shapenet import ShapeNetDataset
from data_utils.scanobjectnn import ScanObjectNN
from backbones.all_backbones import Backbone
from poolings.all_poolings import Pooling
from poolings.coupled_pooling import CoupledPooling
from classifier import Classifier

import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import json
import time

base_seed = 2054
batch_size = 32
num_points_per_set = 1024
min_lr = 0.001

# Optimizer hyperparameters
momentum = 0.9
weight_decay = 1e-4

DATASETS = {
    'modelnet': ModelNet40,
    'shapenet': ShapeNetDataset,
    'scanobjectnn': ScanObjectNN
}

def train_test(backbone_type, pooling_type, dataset='modelnet', dataset_size=0.99, experiment_id=0, optimizer='adam', backbone_args={}, pooling_args={}, gpu_index=0, learning_rate=1e-2, num_epochs=300, early_stopping_patience = 100, scheduler='cos'):
    
    print('learning rate is: {}'.format(learning_rate))
    print('number of epochs is: {}'.format(num_epochs))
    print('early_stopping_patience: {}'.format(early_stopping_patience))
    
    device = f'cuda:{gpu_index}'

    random_seed = int(base_seed + experiment_id)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    coupled = (type(pooling_type) in [list, tuple])
    if coupled:
        poolings = pooling_type
        pooling_type = '_'.join(poolings)   

    # create results directory if it doesn't exist
    backbone_config = "_".join([str(v) for v in backbone_args.values()])
    pooling_config = "_".join([str(v) for v in pooling_args.values()])
    results_dir = f"./results/{dataset}-{dataset_size}/{backbone_type}_{pooling_type}_{experiment_id}/{backbone_config}/{pooling_config}"
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'backbone_args.json'), 'w') as f:
        json.dump(backbone_args, f, indent=2)
    with open(os.path.join(results_dir, 'pooling_args.json'), 'w') as f:
        json.dump(pooling_args, f, indent=2)

    # get the datasets
    base_dataset = DATASETS[dataset]
    num_classes = base_dataset.num_classes
    print(f"Number of classes: {num_classes}")

    phases = ['train', 'valid', 'test']
    dataset = {}
    for phase in phases:
        dataset[phase] = base_dataset(num_points_per_set, partition=phase, dataset_size=dataset_size, seed=random_seed)

    print(f"Size of training dataset: {len(dataset['train'])}")

    # create the dataloaders
    loader = {}
    for phase in phases:
        if phase == 'train':
            shuffle = True
        else:
            shuffle = False
        loader[phase] = DataLoader(dataset[phase], batch_size=batch_size, shuffle=shuffle)

    # create the modules
    backbone = Backbone(backbone_type=backbone_type, **backbone_args)
    if coupled:
        pooling = CoupledPooling(poolings, backbone.d_out, pooling_args)
    else:
        pooling = Pooling(pooling_type=pooling_type, d_in=backbone.d_out, **pooling_args)
    classifier = Classifier(pooling.d_out, num_classes)

    backbone.to(device)
    pooling.to(device)
    classifier.to(device)

    # start training
    criterion = nn.CrossEntropyLoss()

    params = []
    if list(backbone.parameters()):
        params += list(backbone.parameters())
    if list(pooling.parameters()):
        params += list(pooling.parameters())
    if list(classifier.parameters()):
        params += list(classifier.parameters())
    
    if optimizer == 'adam':
        optim = Adam(params, lr=learning_rate, weight_decay=weight_decay)
    else:
        optim = SGD(params, lr=100*learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    if scheduler=='cos':
        scheduler = CosineAnnealingLR(optim, num_epochs, eta_min=min_lr)
    else:
        scheduler = StepLR(optim, step_size=20, gamma=0.7)
        

    epochMetrics = defaultdict(list)
    early_stopping_reached = False
    early_stopping_counter = 0
    best_valid_loss = float('inf')
    save_results = True
    final_results = {}


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
            time_=[]
            

            for i, data in enumerate(loader[phase]):

                # zero the parameter gradients
                optim.zero_grad()

                x, y = data

                x = x.to(device).to(torch.float)
                y = y.to(device).squeeze()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # pass the sets through the backbone and pooling
                    z = backbone(x)
                    # print(f"Backbone: {z.shape}") # 32, 1024, 3
                    v = pooling(z)
                    # print(f"Pool output: {v.shape}") # 32, 3
                    logits = classifier(v)
                    #print(f"Logits: {logits.shape}")
                    # print(f"y: {y.shape}")
                    loss = criterion(logits, y)

                    acc = (1. * (torch.argmax(logits, dim=1) == y)).mean().item()

                    # backpropogation only in training phase
                    if phase == 'train':
                        # Backward pass
                        tic=time.time()
                        loss.backward(retain_graph=True)
                        # 1-step gradient descent
                        optim.step()
                        toc=time.time()
                        time_.append(toc-tic)
                        mean_time=np.mean(time_)

                # save losses and accuracies
                loss_.append(loss.item())
                acc_.append(acc)
                
             
            mean_loss = np.mean(loss_)
            mean_acc = np.mean(acc_)
            

            # Early stopping logic
            if phase == "valid":
                if mean_loss >= best_valid_loss:
                    early_stopping_counter += 1
                    save_results = False
                    if early_stopping_counter >= early_stopping_patience:
                        early_stopping_reached = True
                    
                    print(f"Early stopping counter now {early_stopping_counter}. Early stopping reached: {early_stopping_reached}.")
                else:
                    early_stopping_counter = 0
                    save_results = True
                    best_valid_loss = mean_loss
            elif phase == 'test':
                if save_results:
                    final_results = {
                        'loss': mean_loss,
                        'acc': mean_acc
                    }
                    torch.save(backbone.state_dict(), os.path.join(results_dir, f"backbone_{random_seed}.pth"))
                    torch.save(pooling.state_dict(), os.path.join(results_dir, f"pooling_{random_seed}.pth"))
                    torch.save(classifier.state_dict(), os.path.join(results_dir, f"classifier_{random_seed}.pth"))


            epochMetrics[f'{phase}_loss'].append(mean_loss)
            epochMetrics[f'{phase}_acc'].append(mean_acc)
            epochMetrics['backward_time'].append(np.mean(mean_time))
            

        scheduler.step()
        
        # print(epochMetrics)
        with open(os.path.join(results_dir, f"epoch_metrics_{random_seed}.json"), 'w') as f:
            json.dump(epochMetrics, f, indent=2)
        
        with open(os.path.join(results_dir, f"final_results_{random_seed}.json"), 'w') as f:
            json.dump(final_results, f, indent=2)

        if early_stopping_reached:
            break

    return epochMetrics