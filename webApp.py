import streamlit as st
from PIL import Image
import SimpleITK as sitk
import prediction

# -- Page setting
st.set_page_config(
    page_title="AD Classifier",
)

# -- Sidebar setting
sample = st.sidebar.radio("Pick a sample", ("Upload Sample", "Sample 1", "Sample 2", "Sample 3"))
if sample == "Sample 1":
    path = './data/0.nii'
elif sample == "Sample 2":
    path = './data/1.nii'
elif sample == "Sample 3":
    path = './data/2.nii'
else:
    path = ''
data_path = st.sidebar.text_input("Source root for testing data ?")
pre_select = st.sidebar.selectbox(
    "Which pretreatment would you like to conduct ?",
    ("binaryzation", "AD", "MCI")
)
model_select = st.sidebar.selectbox(
    "Which model would you like to choose ?",
    ("2D model", "3D model", "resnet")
)

# -- Prediction
classify = False
st.title('AD CLASSIFICATION')
st.markdown("""
 * Use the menu at left to select data and model for classification
 * Your result will appear below
""")

st.subheader("Prediction")
if sample != "Upload Sample":
    data_path = path
if not data_path:
    st.write("Input source root for testing data to continue")
else:
    st.write("Preview for nii serial images of the patient")
    img = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(img)
    x = st.slider('Input image serial number', max_value=169)
    res = data[x, :, :]
    res = Image.fromarray(res)
    res = res.convert('RGB')
    st.image(res, use_column_width=256)
    classify = st.button("CLASSIFY")


def num_to_result(num):
    numbers = {
        1: "AD identified",
        0: "CN identified",
        2: "MCI identified"
    }
    return numbers.get(num, None)


if classify:
    result = prediction.predictData(data_path, pre_select, model_select)
    st.write(num_to_result(result))
else:
    st.write('Standing by')
with st.beta_expander("See notes"):
    st.markdown("""
 * Tongji University School of Software Digital Media Orientation Professional Comprehensive Project(2021)
 * Powered by Streamlit.
See also:
 * [Report](https://github.com/AiGaoShiBOY/AD-Recognization)
""")

st.subheader("About this app")
st.markdown("""
This app is based on data from Alzheimer’s Disease Neuroimaging Initiative downloaded from
Image and Data Archive at http://adni.loni.usc.edu/ .
You can see how this works in the [project document](https://github.com/AiGaoShiBOY/AD-Recognization#readme) or 
[see the code](https://github.com/AiGaoShiBOY/AD-Recognization).
""")

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
