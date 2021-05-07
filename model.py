import torch
import torch.nn as nn
from torchvision import models



class COVNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layer_list = list(model.children())[1:-2]
        self.pretrained_model = nn.Sequential(*layer_list)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(2048, n_classes)
        self.n_classes = n_classes

    def forward(self, x):
        x = self.conv1(x)
        features = self.pretrained_model(x)
        pooled_features = self.pooling_layer(features)

        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        #flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]

        output = self.classifer(pooled_features)
        return output