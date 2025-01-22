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
st.set_page_config(page_title="Flower Image Classification", page_icon="🌸", layout="wide")

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
    st.markdown("**About the Model:** This model classifies flowers into 5 categories: daisy, dandelion, roses, sunflowers, and tulips.")
    
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
                <div class="feature-hover">Fast Classification
                    <span class="tooltip-text">Get predictions in seconds. Enjoy a sleek and modern design.</span>
                </div>
            </li>
            <li>
                <div class="feature-hover">Highly Accurate
                    <span class="tooltip-text">Model accuracy is high, making predictions with confidence.</span>
                </div>
            </li>
        </ul>
    """, unsafe_allow_html=True)

    # Contact information
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Contact us at: [**Kelompok 4**](https://www.linkedin.com/in/het-patel-8b110525a/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')

# Flower class names
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Load model
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model("final_model1.h5")  # Replace with your model path
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# Main title with cool text effect
st.markdown("""
    <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
    🌸 Flower Image Classification
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

st.header("Upload Gambar Disini dan Dapatkan Prediksinya!")

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
image_file = st.file_uploader("🌄 Upload an image", type=["jpg", "png"], key="file_uploader")

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

# Additional flower information
st.markdown("""
### **Jenis Bunga**:
- **Daisy**: Bunga umum dengan cakram tengah dan kelopak putih.
- **Dandelion**: Bunga kuning yang berubah menjadi bola bulu ketika matang.
- **Mawar**: Bunga populer yang dikenal dengan warna-warnanya yang cerah dan harum.
- **Bunga Matahari**: Bunga besar berwarna kuning yang dikenal dengan ukurannya dan heliotropisme.
- **Tulip**: Bunga berwarna-warni dengan bentuk yang halus dan bulat.
""", unsafe_allow_html=True)

# Data for flower classification performance
data = {
    "Class": ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips'],
    "Accuracy": [0.92, 0.88, 0.89, 0.93, 0.85],  # Example values, replace with real metrics
    "Precision": [0.90, 0.85, 0.87, 0.92, 0.84]  # Example values, replace with real metrics
}

df = pd.DataFrame(data)

# Stylish DataFrame with 5 rows
st.markdown("### Flower Classification Performance")
styled_table = df.style.background_gradient(cmap="coolwarm", subset=['Accuracy', 'Precision'])
st.dataframe(styled_table, height=400)
