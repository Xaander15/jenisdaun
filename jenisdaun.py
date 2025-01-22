import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf
import time

# Streamlit page configuration
st.set_page_config(page_title="Leaf Image Classification", page_icon="🌿", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🌿 Leaf Classifier")
    st.markdown("Upload a leaf image to classify it into **blimbing**, **jeruk**, or **kemangi**.")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("keras_model.h5", compile=False)  # Ensure model path is correct
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class names (ensure this matches your `labels.txt`)
class_names = ['blimbing', 'jeruk', 'kemangi']

# Function to preprocess the image
def preprocess_image(image):
    size = (224, 224)  # Match model input size
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array.reshape(1, 224, 224, 3)

# Title
st.title("🌱 Leaf Image Classification")
st.subheader("Upload a leaf image and get its classification!")

# Upload image
image_file = st.file_uploader("Upload an image (JPG/PNG):", type=["jpg", "png"])

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Image 🧠"):
        with st.spinner("Classifying..."):
            # Preprocess and predict
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            index = np.argmax(predictions)
            confidence = predictions[0][index]
            class_name = class_names[index]

            # Display result
            if confidence > 0.6:  # Confidence threshold
                st.success(f"Prediction: {class_name.capitalize()} (Confidence: {confidence*100:.2f}%)")
            else:
                st.warning("Prediction confidence is too low. Unable to classify.")

# Footer
st.markdown("---")
st.markdown("Created by **Kelompok 4** - 2025")
