import os
import torch
import torch.nn as nn
import csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.autograd import Variable
from dataset import ADSet
import torch.nn.functional as F
import numpy as np
import time
import model
import utils

batch_size = 100
train_set = ADSet()

validation_split = .2
shuffle_dataset = True
random_seed = 42
set_size = len(train_set)
print("set_size:{}".format(set_size))
indices = list(range(set_size))
split = int(np.floor(validation_split * set_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                sampler=valid_sampler)

for i,(image,label) in enumerate(train_loader):
    print(label)
