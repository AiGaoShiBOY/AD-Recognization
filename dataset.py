import readData

from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
import SimpleITK as sitk

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class ADSet(Dataset):
    def __init__(self):
        self.path_list, self.label_list = readData.get_data_list()
        self.transform = train_transform()

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = self.path_list[index]
        img = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(img)
        if self.transform:
            img = self.transform(img)

        label = self.label_list[index]

        return image, label



