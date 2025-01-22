import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow.keras.utils import img_to_array, load_img
import tensorflow as tf
import requests
import time
import streamlit_lottie as st_lottie

# Streamlit page configuration
st.set_page_config(page_title="Leaf Classification", page_icon="🍃", layout="wide")

# Load Lottie animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie Animation
lottie_url = "https://lottie.host/de06d967-8825-499e-aa8c-a88dd15e1a08/dH2OtlPb3c.json"
lottie_animation = load_lottie_url(lottie_url)

# Sidebar with unique elements
with st.sidebar:
    st_lottie.st_lottie(lottie_animation, height=200, width=200, key="lottie_animation")
    st.markdown("<h2 style='color: #007bff;'>Explore the App!</h2>", unsafe_allow_html=True)
    st.markdown("**About the Model:** This model classifies leaves into 3 categories: daun belimbing, daun jeruk, and daun kemangi.")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Contact us at: [**Kelompok 4**](https://www.linkedin.com)')

# Load class names from label.txt
def load_class_names(file_path):
    with open(file_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

class_names = load_class_names("labels.txt")

# Load model
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model("keras_model.h5", custom_objects={'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# Main title with cool text effect
st.markdown("""
    <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
    🍃 Leaf Classification
    </h1>
    <style>
    @keyframes glow {
        0% {
            text-shadow: 0 0 10px #9b59b6, 0 0 20px #007bff, 0 0 30px #007bff, 0 0 40px #9b59b6;
        }
        100% {
            text-shadow: 0 0 20px #8e44ad, 0 0 30px #007bff, 0 0 40px #007bff, 0 0 50px #8e44ad;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.header("Upload Gambar Daun dan Dapatkan Prediksinya!")

# Image loading function
def load_image(filename):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img
    
# Create folder for images if not exist
if not os.path.exists('./images'):
    os.makedirs('./images')

# Upload image section with fancy file uploader
image_file = st.file_uploader("🍂 Upload an image", type=["jpg", "png"], key="file_uploader")

if image_file is not None:
    if st.button("Classify Image 🧠", key="classify_button"):
        img_path = f"./images/{image_file.name}"
        with open(img_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        image = Image.open(img_path)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        img_to_predict = load_image(img_path)

        # Progress spinner
        with st.spinner('🔍 Classifying image...'):
            time.sleep(2)
            try:
                predictions = model.predict(img_to_predict)
                predicted_class = np.argmax(predictions, axis=-1)
                confidence = np.max(predictions)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                predictions = None

        # Threshold and result display
        if predictions is not None:
            confidence_threshold = 0.60  # Confidence threshold 60%

            if confidence < confidence_threshold:
                result = f"Prediction: Not a recognized leaf (Confidence: {confidence*100:.2f}%)"
            else:
                result = f"Prediction: {class_names[predicted_class[0]]} with {confidence*100:.2f}% confidence"

            st.success(result)

        os.remove(img_path)

# Additional information about leaves
st.markdown("""
### **Jenis Daun**:
- **Daun Belimbing**: Ciri khas memiliki tepi yang bergerigi halus.
- **Daun Jeruk**: Berwarna hijau mengkilap dan sering digunakan sebagai aroma masakan.
- **Daun Kemangi**: Bertekstur lembut dan memiliki aroma khas yang kuat.
""", unsafe_allow_html=True)
