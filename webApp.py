import streamlit as st
import time
import numpy as np
import SimpleITK as sitk

st.set_page_config(
    page_title="AD Classifier",
)
file_data = st.file_uploader("Select an image", type=["jpg", "jpeg", "nii"])
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

if file_data is not None:
    with st.spinner('Classifying...'):
        # load the image from uploader; fix rotation for iOS devices if necessary
        img = file_data
        st.write('## Your Image')
        st.image(img, width=200)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
