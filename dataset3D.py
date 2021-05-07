import readData

from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
import SimpleITK as sitk
import scipy.ndimage
import numpy as np


class ADSet(Dataset):
    def __init__(self, transform=None):
        self.path_list, self.label_list = readData.get_data_list()
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = self.path_list[index]
        img = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(img)
        resampled_img = scipy.ndimage.zoom(image, (64/image.shape[0], 128/image.shape[1], 128 / image.shape[2]), order=3)
        if self.transform is not None:
            resampled_img = self.transform(resampled_img)

        label = self.label_list[index]

        return resampled_img, label




