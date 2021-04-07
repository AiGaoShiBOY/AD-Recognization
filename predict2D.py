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

model = torchvision.models.resnet50(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# load the model
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 3),
    nn.LogSoftmax(dim=1)
)
# pre-trained model
path = "ad_classify_model_1.model"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint)
model.eval()

# open output file
wf = open('result.txt', 'w+')


def predict_image(image_path):
    image = Image.open(image_path)
    if image.mode == 'L':
        image = image.convert('RGB')

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Preprocess the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)

    index = output.data.numpy().argmax()

    return index


if __name__ == "__main__":

    imagepath = "testData/"
    path = [
        os.path.join(x)
        for x in os.listdir("testData/") if x[0] != '.'
    ]

    # run prediction function and obtain predicted class index
    for j in range(len(path)):
        image_name = [
            os.path.join(x)
            for x in os.listdir(imagepath + path[j]) if x[0] != '.'
        ]
        class_list = []
        for i in range(len(image_name)):
            index = predict_image(imagepath + path[j] + "/" + image_name[i])
            class_list.append(index)
            # print("Predicted Class ", index , "image name" , imagepath+path[j]+"/"+image_name[i])
        p_num = class_list.count(1)
        n_num = class_list.count(0)
        c_num = class_list.count(2)
        print(str(n_num) + '  ' + str(p_num) + '  ' + str(c_num))
        if p_num > 0.4 * len(class_list):
            res = 'AD identified'
            print(path[j] + ',AD')
            wf.write(path[j] + ',AD\n')
        elif c_num > 0.2 * len(class_list):
            res = 'CN identified'
            print(path[j] + ',CN')
            wf.write(path[j] + ',CN\n')
        else:
            res = 'MCI'
            print(path[j] + ',MCI')
            wf.write(path[j] + ',MCI\n')

wf.close()
