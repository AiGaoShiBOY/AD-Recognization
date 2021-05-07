import readData

from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
import nibabel as nib
import numpy as np
from PIL import Image
import random

AX_F = 0.32
COR_F = 0.56
SAG_F =  0.56
NON_AX = (1, 2)
NON_COR = (0, 2)
NON_SAG = (0, 1)
AX_SCETION = "[slice_i, :, :]"
COR_SCETION = "[:, slice_i, :]"
SAG_SCETION = "[:, :, slice_i]"

class SliceSet2D(Dataset):
    def __init__(self, transform=None):
        self.path_list, self.label_list = readData.get_data_list()
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        path = self.path_list[index]
        image = nib.load(path)

        label = self.label_list[index]
        samples = []

        AXimageList = axRandomSlice(image)
        CORimageList = corRandomSlice(image)
        SAGimageList = sagRandomSlice(image)

        for img2DList in (AXimageList, CORimageList, SAGimageList):
            for image2D in img2DList:
                image2D = Image.fromarray(image2D.astype(np.uint8), 'RGB')
                if self.transform:
                    image2D = self.transform(image2D)
                sample = {'image': image2D, 'label': label}
                samples.append(sample)
        random.shuffle(samples)
        return samples


def getRandomSlice(image_array, mean_direc, fraction, section, step = 2):
    mean_array = np.ndarray.mean(image_array, axis = mean_direc)
    first_p = list(mean_array).index(list(filter(lambda x: x>0, mean_array))[0])
    last_p = list(mean_array).index(list(filter(lambda x: x>0, mean_array))[-1])
    slice_p = int(round(first_p + (last_p - first_p)*fraction))
    slice_2Dimgs = []
    slice_select_0 = None
    slice_select_1 = None
    slice_select_2 = None

    randomShift = random.randint(-18, 18)
    slice_p = slice_p + randomShift
    i = 0
    for slice_i in range(slice_p-step, slice_p+step+1, step):
        slice_select = eval("image_array"+section)/1500.0*255
        if i == 0:
            slice_select_0 = slice_select
        if i == 1:
            slice_select_1 = slice_select
        if i == 2:
            slice_select_2 = slice_select
        i += 1
    slice_2Dimg = np.stack((slice_select_0, slice_select_1, slice_select_2), axis = 2)
    slice_2Dimgs.append(slice_2Dimg)
    return slice_2Dimgs

def axRandomSlice(image):
    image_array = np.array(image.get_data())
    return getRandomSlice(image_array, NON_AX, AX_F, AX_SCETION)


def corRandomSlice(image):
    image_array = np.array(image.get_data())
    return getRandomSlice(image_array, NON_COR, COR_F, COR_SCETION)



def sagRandomSlice(image):
    image_array = np.array(image.get_data())
    return getRandomSlice(image_array, NON_SAG, SAG_F, SAG_SCETION)




