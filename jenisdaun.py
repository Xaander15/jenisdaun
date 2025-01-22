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
from tensorflow.keras.models import model_from_json

# Streamlit page configuration
st.set_page_config(page_title="Leaf Image Classification", page_icon="🌿", layout="wide")

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
    st.markdown("**About the Model:** This model classifies leaves into 3 categories: blimbing, jeruk, and kemangi.")

    # Features section
    st.markdown("Fast Classification: Get predictions in seconds!")
    st.markdown("Highly Accurate: Model accuracy is high, making predictions with confidence.")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Contact us at: [**Kelompok 4**](https://www.linkedin.com/in/het-patel-8b110525a/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')

# Load and modify model function
@st.cache_resource
def load_and_modify_model():
    try:
        # Load model directly
        model = tf.keras.models.load_model("keras_Model.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model directly: {e}")
        try:
        
            # Modify problematic layers
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                    layer_config = layer.get_config()
                    if "groups" in layer_config:
                        del layer_config["groups"]
                    new_layer = tf.keras.layers.DepthwiseConv2D(**layer_config)
                    model = tf.keras.models.clone_model(
                        model,
                        clone_function=lambda l: new_layer if l == layer else l,
                    )
            model.compile()
        except Exception as ex:
            st.error(f"Error modifying model: {ex}")
            return None
    return model

model = load_and_modify_model()

# Leaf class names
class_names = ['blimbing', 'jeruk', 'kemangi']

# Main title
st.markdown("<h1 style='text-align:center; color: #007bff;'>🌿 Leaf Image Classification</h1>", unsafe_allow_html=True)
st.header("Upload Gambar Daun Disini dan Dapatkan Prediksinya!")

# Image loading function
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # Match model's input size
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img = img / 127.5 - 1  # Normalize based on model preprocessing
    return img

# Create folder for images if not exist
if not os.path.exists('./images'):
    os.makedirs('./images')

# Upload image section
image_file = st.file_uploader("🌞 Upload an image", type=["jpg", "png"], key="file_uploader")

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
            confidence_threshold = 0.60  # Confidence threshold

            if confidence < confidence_threshold:
                result = f"Prediction: Not a recognized leaf (Confidence: {confidence*100:.2f}%)"
            else:
                result = f"Prediction: {class_names[predicted_class[0]]} with {confidence*100:.2f}% confidence"

            st.success(result)

        os.remove(img_path)

# Additional leaf information
st.markdown("### **Jenis Daun**:\n- **Blimbing**: Daun dengan bentuk khas menyerupai bintang.\n- **Jeruk**: Daun hijau yang sering digunakan sebagai rempah aromatik.\n- **Kemangi**: Daun kecil beraroma khas, sering digunakan dalam masakan tradisional.", unsafe_allow_html=True)
