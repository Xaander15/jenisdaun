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
st.set_page_config(page_title="Leaf Classification", page_icon="🌿", layout="wide")

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
    st.markdown("**About the Model:** This leaf classifier uses a convolutional neural network trained on thousands of leaf images.")
    
    # Features section with hover effect
    st.markdown(""" 
        <style>
            .feature-hover {
                position: relative;
                display: inline-block;
                color: #007bff;
                cursor: pointer;
            }

            .feature-hover .tooltip-text {
                visibility: hidden;
                width: 200px;
                background-color: #333;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 100%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }

            .feature-hover:hover .tooltip-text {
                visibility: visible;
                opacity: 1;
            }
        </style>

        <ul>
            <li>
                <div class="feature-hover">Fast Classification(Cool)
                    <span class="tooltip-text">Get predictions in seconds.Enjoy a sleek and modern design.</span>
                </div>
            </li>
            <li>
                <div class="feature-hover">Highly Accurate
                    <span class="tooltip-text">Model accuracy is up to 92%.</span>
                </div>
            </li>
        </ul>
    """, unsafe_allow_html=True)

    # Contact information
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Contact us at: [**Kelompok 4**](https://www.linkedin.com/in/het-patel-8b110525a/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')

# Leaf class names
class_names = [
    "blimbing", "jeruk", "kemangi"
]

# Load model
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("final_model1.h5")
    return model

model = load_my_model()

# Main title with cool text effect
st.markdown("""
    <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
    🌿 Leaf Classification App by Kelompok 4
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

st.header("Upload Gambar Daun Disini dan Dapatkan Prediksinya!")

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
image_file = st.file_uploader("🌿 Upload an image", type=["jpg", "png"], key="file_uploader")

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
            predictions = model.predict(img_to_predict)
            predicted_class = np.argmax(predictions, axis=-1)
            confidence = np.max(predictions)

          # Threshold and result display
        if predictions is not None:
            confidence_threshold = 0.60  # Increased confidence threshold to 60%

            if confidence < confidence_threshold:
                result = f"Prediction: Not a recognized flower (Confidence: {confidence*100:.2f}%)"
            else:
                result = f"Prediction: {class_names[predicted_class[0]]} with {confidence*100:.2f}% confidence"

            st.success(result)

        os.remove(img_path)

# Add unique progress bar for better interactivity
if st.button("Reload App"):
    st.progress(100)

# Additional leaf information
st.markdown(""" 
### **Jenis Daun**:
- <span title="Daun dari tanaman blimbing.">**blimbing**</span>
- <span title="Daun dari tanaman jeruk.">**jeruk**</span>
- <span title="Daun dari tanaman kemangi.">**kemangi**</span>
""", unsafe_allow_html=True)

# Data for leaf performance
data = {
    "Leaf Type": class_names,
    "Accuracy": [0.88, 0.85, 0.90],
    "Precision": [0.86, 0.84, 0.89]
}
df = pd.DataFrame(data)

# Stylish DataFrame
st.markdown("### Leaf Accuracy and Precision")
styled_table = df.style.background_gradient(cmap="coolwarm", subset=['Accuracy', 'Precision'])
st.dataframe(styled_table, height=400)
