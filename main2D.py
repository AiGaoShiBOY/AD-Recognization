import argparse
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision.models as models
from torch.nn import Linear, CrossEntropyLoss, Softmax
from torch.optim import Adam,SGD
import os

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image

from Custom_transform import CustomResize
from Custom_transform import CustomToTensor

import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
from collections import Counter

from dataset2DRandomslice import SliceSet2D

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Configurations")

parser.add_argument("--load", default = False, type = bool,
                    help="Load saved network weights.")
parser.add_argument("--save", default="AlexNet",
                    help="Save network weights.")
parser.add_argument("--augmentation", default=True, type=bool,
                    help="Save network weights.")
parser.add_argument("--epochs", default=50, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=3e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--estop", default=1e-2, type=float,
                    help="Early stopping criteria on the development set. (default=1e-2)")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")


# feel free to add more arguments as you need


def main(options):

    model_dir = os.path.join('experiments', 'models')
    trg_size = (224, 224)
    transformations = transforms.Compose([CustomResize(trg_size),
                                          CustomToTensor()
                                        ])
    batch_size = options.batch_size
    train_set = SliceSet2D(transform=transformations)

    validation_split = .4
    shuffle_dataset = True
    random_seed = 42
    set_size = len(train_set)
    indices = list(range(set_size))

    split = int(np.floor(validation_split * set_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print("Set Size: {}|Train Size: {}| Validation Size: {}".format(len(indices), len(train_indices), len(val_indices)))
    print("Model will be saved in \" ./experiments/model\". Start training...")
    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                    sampler=valid_sampler)


    # Initiate the model
    model = models.resnet18(pretrained=True)
    model.fc = Linear(in_features=512, out_features=3)
    for para in list(model.parameters())[:-2]:
        para.requires_grad = False
    optimizer = Adam(params=[model.fc.weight, model.fc.bias], lr=3e-4)
    print('the training layer is:')
    for name, param in model.named_parameters():  # 查看可优化的参数有哪些
        if param.requires_grad:
            print(name)
    # model.load_state_dict(torch.load(options.load))

    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    # Binary cross-entropy loss
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    lr = options.learning_rate
    optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()), lr)

    best_accuracy = float("-inf")



    for epoch_i in range(options.epochs):

        logging.info("At {0}-th epoch.".format(epoch_i))
        train_loss, correct_cnt = train(model, train_loader, criterion, optimizer)
        # each instance in one batch has 3 views
        train_avg_loss = train_loss / len(train_indices)*3/options.batch_size
        train_avg_acu = float(correct_cnt) / (len(train_indices) * 3)
        logging.info(
            "Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data, epoch_i))
        logging.info("Average training accuracy is {0:.5f} at the end of epoch {1}".format(train_avg_acu, epoch_i))

        with torch.no_grad():
            correct_cnt = validate(model, validation_loader)
            dev_avg_acu = float(correct_cnt) / len(val_indices)
            logging.info("Average validation accuracy is {0:.5f} at the end of epoch {1}".format(dev_avg_acu, epoch_i))


        if dev_avg_acu > best_accuracy:
            best_accuracy = dev_avg_acu
            torch.save(model, os.path.join('experiments', 'models/mymodel.pth'))


def train(model, train_loader, criterion, optimizer):
    # main training loop
    train_loss = 0.0
    correct_cnt = 0.0
    model.train()
    for it, train_data in enumerate(train_loader):
        # 一个loader将返回3个dictionary，这三个值将全部输入网络
        for data_dic in train_data:
            if torch.cuda.is_available():
                imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label']).cuda()
            else:
                imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])
            integer_encoded = labels.data.cpu().numpy()
            ground_truth = Variable(torch.from_numpy(integer_encoded)).long()
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda()
            train_output = model(imgs)
            _, predict = train_output.topk(1)
            loss = criterion(train_output, ground_truth)
            train_loss += loss
            correct_this_batch = (predict.squeeze(1) == ground_truth).sum().float()
            correct_cnt += correct_this_batch
            accuracy = float(correct_this_batch) / len(ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return train_loss, correct_cnt



def validate(model, val_loader):
    # validation -- this is a crude estimation because there might be some paddings at the end
    correct_cnt = 0.0
    model.eval()
    for it, test_data in enumerate(val_loader):
        vote = []
        for data_dic in test_data:
            if torch.cuda.is_available():
                imgs, labels = Variable(data_dic['image']).cuda(), Variable(data_dic['label'],).cuda()
            else:
                imgs, labels = Variable(data_dic['image']), Variable(data_dic['label'])
            test_output = model(imgs)
            _, predict = test_output.topk(1)
            vote.append(predict)

        vote = torch.cat(vote, 1)
        final_vote, _ = torch.mode(vote, 1)
        ground_truth = test_data[0]['label']
        correct_this_batch = (final_vote.cpu().data == ground_truth).sum()
        correct_cnt += correct_this_batch
        accuracy = float(correct_this_batch) / len(ground_truth)

    return correct_cnt




def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)