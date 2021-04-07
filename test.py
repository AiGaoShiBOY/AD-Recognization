import SimpleITK as sitk
from PIL import Image
import skimage.io as io


# path = 'F:\\SSE2021Spring\\专业综合项目\\AD-Recognization-main\\AD-Recognization-main\\ADNI\\018_S_0633\\MPR____N3__Scaled\\2006-07-17_11_06_31.0\\S16900\\1.png'
# im = Image.open(path)
# im.show()

# 数据所在路径

path = 'F:\\SSE2021Spring\\专业综合项目\\AD-Recognization-main\\AD-Recognization-main\\ADNI\\018_S_0633\\MPR____N3__Scaled\\2006-07-17_11_06_31.0\\S16900\\ADNI_018_S_0633_MR_MPR____N3__Scaled_Br_20070101223315779_S16900_I35039.nii'
img = sitk.ReadImage(path)
data = sitk.GetArrayFromImage(img)
for i in range(data.shape[0]):
    io.imshow(data[i, :, :], cmap='gray')
    print(i)
    io.show()
