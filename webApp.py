import streamlit as st
from PIL import Image
import time
import numpy as np
import SimpleITK as sitk

st.set_page_config(
    page_title="AD Classifier",
)
sample_select = st.sidebar.selectbox(
    "How would you like your patient to be ?",
    ("CN", "AD", "MCI")
)
data_path = st.sidebar.text_input("Source root for testing data ?")
pre_select = st.sidebar.selectbox(
    "Which pretreatment would you like to conduct ?",
    ("binaryzation", "AD", "MCI")
)
st.write("## Pretreatment")
file_data = st.file_uploader("Select an image", type=["jpg", "jpeg", "nii"])

path = 'F:\\SSE2021Spring\\AD-2D\\AD-Recognization-main\\AD-Recognization-main\\ADNI\\018_S_0633\\MPR____N3__Scaled\\2006-07-17_11_06_31.0\\S16900\\ADNI_018_S_0633_MR_MPR____N3__Scaled_Br_20070101223315779_S16900_I35039.nii'
st.write(file_data)
img = sitk.ReadImage(data_path)
data = sitk.GetArrayFromImage(img)
x = st.slider('Input image serial number', max_value=169)
res = data[x, :, :]
res = Image.fromarray(res)
res = res.convert('RGB')
st.image(res, use_column_width=256)

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
