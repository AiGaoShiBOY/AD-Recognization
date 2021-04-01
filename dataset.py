import readData

from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
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
        label = self.label_list[index]
        return image,label


