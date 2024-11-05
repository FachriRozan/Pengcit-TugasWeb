import streamlit as st
import pandas as pd
import requests
from PIL import Image
import os
st.set_page_config(layout="wide", page_title="APNR App")
st.title("API APNR")

st.markdown(
    """
    <style>
    .stMainBlockContainer, .block-container, .st-emotion-cache-13ln4jf {
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

upload_folder = "images"
os.makedirs(upload_folder, exist_ok=True)  
def send_image_to_api(image_path):
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post("http://127.0.0.1:5000/prediction", files=files)
        if response.status_code == 200:
            return response.json() 
        else:
            st.error(f"Gagal menghubungi API. Status code: {response.status_code}")
            return None

uploaded_image = st.file_uploader("Upload gambar di sini", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Gambar yang di-upload", use_column_width=True)
    
    image_path = os.path.join(upload_folder, uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())


    response_data = send_image_to_api(image_path)
    
    if response_data:
        st.subheader("Respons dari API")
        if isinstance(response_data["data"], list) and isinstance(response_data["data"][0], dict):
            df = pd.DataFrame(response_data["data"])
        else:
            df = pd.DataFrame([response_data["data"]])
        st.table(df)
