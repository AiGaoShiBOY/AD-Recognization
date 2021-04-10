import os
from sched import scheduler

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
import config



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, train_loader, epoch, num_epochs, optimizer, current_lr, log_every=100):
    n_classes = model.n_classes
    metric = torch.nn.CrossEntropyLoss()

    y_probs = np.zeros((0, n_classes), np.float)
    losses, y_trues = [], []
    model.train()

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        label = np.array(label)
        y_trues = np.append(y_trues, label)

    y_trues = np.array(y_trues)
    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
    train_loss_epoch = np.round(np.mean(losses), 4)

    return train_loss_epoch, metric_collects


def evaluate_model(model, val_loader, epoch, num_epochs, current_lr, log_every=20):
    n_classes = model.n_classes
    metric = torch.nn.CrossEntropyLoss()

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    y_probs = np.zeros((0, n_classes), np.float)
    losses, y_trues = [], []

    for i, (image, label) in enumerate(val_loader):

        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        prediction = model.forward(image.float())
        loss = metric(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)
        y_prob = F.softmax(prediction, dim=1)
        y_probs = np.concatenate([y_probs, y_prob.detach().cpu().numpy()])
        label = np.array(label)
        y_trues = np.append(y_trues, label)



    val_loss_epoch = np.round(np.mean(losses), 4)
    y_trues = np.array(y_trues)
    metric_collects = utils.calc_multi_cls_measures(y_probs, y_trues)
    return val_loss_epoch, metric_collects


def main(args):

    print("The parameters are set as follows:")
    print(args)
    # 首先处理文件保存路径
    exp_dir = 'experiments'
    model_dir = os.path.join(exp_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    batch_size = args.batch_size
    train_set = ADSet()

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
    cov_net = model.COVNet(3)

    if torch.cuda.is_available():
        cov_net = cov_net.cuda()

    optimizer = Adam(cov_net.parameters(), lr=args.lr, weight_decay=0.1)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size=3, gamma=args.gamma)

    best_val_loss = float('inf')
    best_val_accu = float(0)

    iteration_change_loss = 0
    t_start_training = time.time()


    ##########################################################################
    #                           Main training loop                           #
    ##########################################################################

    epochs = args.epochs

    for epoch in range(epochs):
        current_lr = get_lr(optimizer)
        t_start = time.time()

        ############################################################
        #  The actual training and validation step for each epoch  #
        ############################################################
        train_loss, train_metric = train_model(
            cov_net, train_loader, epoch, epochs, optimizer,
            current_lr, 100)


        with torch.no_grad():
            val_loss, val_metric = evaluate_model(
                cov_net, validation_loader, epoch, epochs, current_lr)


        ##############################
        #  Adjust the learning rate  #
        ##############################
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        utils.print_epoch_progress(epoch, epochs, train_loss, val_loss, delta, train_metric,
                                   val_metric)
        iteration_change_loss += 1
        print('-' * 60)
        train_acc, val_acc = train_metric['accuracy'], val_metric['accuracy']
        if args.save_model is True:
            file_name = ('train_acc_{}_val_acc_{}_epoch_{}.pth'.
                     format(train_acc, val_acc, epoch))
            torch.save(cov_net, os.path.join(model_dir, file_name))

        if val_acc > best_val_accu:
            best_val_accu = val_acc
            if bool(args.save_model):
                torch.save(cov_net, os.path.join(model_dir, 'best.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == args.patience:
            print(('Early stopping after {0} iterations without the decrease ' +
                  'of the val loss').format(iteration_change_loss))
            break
    t_end_training = time.time()
    print('training took {}s'.
          format(t_end_training - t_start_training))


if __name__ == "__main__":
    args = config.parse_arguments()
    main(args)