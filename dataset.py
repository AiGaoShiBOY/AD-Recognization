import readData

from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
import SimpleITK as sitk
import scipy.ndimage

class ADSet(Dataset):
    def __init__(self):
        self.path_list, self.label_list = readData.get_data_list()

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = self.path_list[index]
        img = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(img)
        Resampledimg = scipy.ndimage.zoom(image, (64/image.shape[0], 128/image.shape[1], 128 / image.shape[2]), order=3)
        Resampledimg = Resampledimg / 255

        label = self.label_list[index]

        return Resampledimg, label



