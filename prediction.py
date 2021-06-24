import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import pandas as pd
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import scipy
import scipy.ndimage


def predictData(data_path, model_select):
    model = tf.keras.models.load_model("./model/model.h5", compile=True)

    itk_img = sitk.ReadImage(data_path)
    img = sitk.GetArrayFromImage(itk_img)

    predict_x = np.zeros([1, 128, 128, 64, 1])
    img = img[21:149, :, :]
    Resampledimg = scipy.ndimage.zoom(img, (64 / img.shape[0], 128 / img.shape[1], 128 / img.shape[2]), order=3)
    for p in range(64):
        for q in range(128):
            for r in range(128):
                predict_x[0, q, r, p, 0] = Resampledimg[p, q, r] / 255

    res = model.predict(predict_x)
    index = np.argmax(res)
    return index
