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
    path = 'F:\\SSE2021Spring\\AD-2D\\AD-Recognization-main\\AD-Recognization-main\\ADNI\\018_S_0633\\MPR____N3__Scaled\\2006-07-17_11_06_31.0\\S16900\\ADNI_018_S_0633_MR_MPR____N3__Scaled_Br_20070101223315779_S16900_I35039.nii'
elif sample == "Sample 2":
    path = 'F:\\SSE2021Spring\\AD-2D\\AD-Recognization-main\\AD-Recognization-main\\ADNI\\018_S_0633\\MPR____N3__Scaled\\2006-07-17_11_06_31.0\\S16900\\ADNI_018_S_0633_MR_MPR____N3__Scaled_Br_20070101223315779_S16900_I35039.nii'
elif sample == "Sample 3":
    path = 'F:\\SSE2021Spring\\AD-2D\\AD-Recognization-main\\AD-Recognization-main\\ADNI\\018_S_0633\\MPR____N3__Scaled\\2006-07-17_11_06_31.0\\S16900\\ADNI_018_S_0633_MR_MPR____N3__Scaled_Br_20070101223315779_S16900_I35039.nii'
else:
    path = ''
data_path = st.sidebar.text_input("Source root for testing data ?")
pre_select = st.sidebar.selectbox(
    "Which pretreatment would you like to conduct ?",
    ("binaryzation", "AD", "MCI")
)
model_select = st.sidebar.selectbox(
    "Which model would you like to choose ?",
    ("CN", "AD", "MCI")
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
        0: "Normal identified",
        1: "AD identified",
        2: "CN identified",
        3: "MCI identified"
    }
    return numbers.get(num, None)


if classify:
    result = prediction.predictData(data_path, pre_select, model_select)
    result = 0
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
