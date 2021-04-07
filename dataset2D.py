import readData

from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import SimpleITK as sitk


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

        img_list = []
        for i in range(0, image.shape[0]):
            tmp = image[i, :, :]
            img_list.append(tmp)

        if self.transform:
            img = self.transform(img)

        label = self.label_list[index]

        return img_list, label
