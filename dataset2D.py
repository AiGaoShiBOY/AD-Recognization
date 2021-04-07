from torchvision.datasets.folder import default_loader
import readData
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
import SimpleITK as sitk


class ADSet(Dataset):
    def __init__(self, transform=None, target_transform=None, loader=default_loader):
        self.path_list, self.label_list = readData.get_data_list()
        imgs = []
        for num in range(len(self.path_list)):
            path = self.path_list[num]
            img = sitk.ReadImage(path)
            image = sitk.GetArrayFromImage(img)
            for i in range(0, image.shape[0]):
                tmp = image[i, :, :]
                tmp = Image.fromarray(tmp)
                imgs.append((tmp, self.label_list[num]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
