import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

from torchvision.transforms import transforms
from .vgg import *

class Dataset():
    
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform      
        self.toPIL = transforms.ToPILImage()
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        x = self.toPIL(self.X[idx])
        x = self.transform(x)
        y = self.target_transform(self.y[idx].item()) if self.target_transform is not None else self.y[idx].item()
        return x, y
        

class RandomRot():
    
    def __init__(self, idx=None):
        self.idx = idx
        self.n_classes = 4

    def __call__(self, x):
        if type(x) == int:
            # This works because ``transforms``` is called before ``target_transform`` in ``torch.Datasets``
            return self.label
        
        rot = [Id, # 0
              transforms.Compose([Transpose2, transforms.RandomHorizontalFlip(1.0), transforms.RandomVerticalFlip(1.0)]), # 90
              transforms.Compose([Transpose1, transforms.RandomHorizontalFlip(1.)]), # 180 
              transforms.Compose([transforms.RandomHorizontalFlip(1.), Transpose2, transforms.RandomVerticalFlip(1.)])   # 270
             ]                                     
        self.label =  np.random.randint(4) if self.idx is None else self.idx
        return rot[self.label](x)
    
    
def compute_logits(dataloader, net):
    net.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets_) in enumerate(dataloader):

            inputs, targets_ = inputs.to(device), targets_.to(device)
            outputs_ = net(inputs).detach().cpu()
            outputs.append(outputs_)
            targets.append(targets_)
    
    return torch.cat(outputs, dim=0), torch.cat(targets, dim=0)

def compute_features(dataloader, net):
    net.eval()
    outputs, targets = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets_) in enumerate(dataloader):

            inputs, targets_ = inputs.to(device), targets_.to(device)
            outputs_ = net(inputs).detach().cpu()
            outputs.append(net.fmaps)
            targets.append(targets_)
    
    return torch.cat(outputs, dim=0), torch.cat(targets, dim=0)


def compute_scores(logits):
    
    softmax = torch.softmax(logits, dim=2)
    
    sum_softmax =  softmax[0, :,  0]
    sum_softmax += softmax[1, :,  1]
    sum_softmax += softmax[2, :,  2]
    sum_softmax += softmax[3, :,  3]
    
    return -sum_softmax

def compute_scores_ica(ica_features):   
    norm = np.linalg.norm(ica_features, ord=2.0, axis=2)
    return norm.mean(0)