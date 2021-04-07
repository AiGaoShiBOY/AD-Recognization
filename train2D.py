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
from dataset2D import ADSet
import numpy as np
import time

# Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

batch_size = 16

train_set = ADSet(transform=train_transform)

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

# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

# Create model, optimizer and loss function
model = torchvision.models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)

if cuda_avail:
    model.cuda()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_models(epoch):
    torch.save(model.state_dict(), "ad_classify_model_{}.model".format(epoch))
    print("Checkpoint saved")


def train(num_epochs):
    # save best accuracy
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            # print("train i:{} print labels:{}".format(i , labels))
            # Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            # Clear all accumulated gradients(梯度)
            optimizer.zero_grad()
            # Predict classes using images from the train sets
            outputs = model(images)
            total += labels.size(0)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate(反向传播) the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)

            train_acc += torch.sum(prediction == labels.data)

        # Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        # Compute the average acc and loss over all training images
        train_acc = train_acc / total
        train_loss = train_loss / total

        # Evaluate on the test set
        test_acc = test()

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,
                                                                                        test_acc))


def test():
    model.eval()
    test_acc = 0.0
    test_total = 0
    for i, (images, labels) in enumerate(validation_loader):

        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        # Predict classes using images from the test set
        outputs = model(images)
        test_total += labels.size(0)
        _, prediction = torch.max(outputs.data, 1)

        test_acc += torch.sum(prediction == labels.data)

    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / test_total
    return test_acc


if __name__ == "__main__":
    train(200)
