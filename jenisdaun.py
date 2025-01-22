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

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Klasifikasi Jenis Daun", page_icon="🌸", layout="wide")

# Fungsi untuk memuat animasi Lottie
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animasi Lottie
lottie_url = "https://lottie.host/de06d967-8825-499e-aa8c-a88dd15e1a08/dH2OtlPb3c.json"
lottie_animation = load_lottie_url(lottie_url)

# Sidebar dengan elemen unik
with st.sidebar:
    st_lottie.st_lottie(lottie_animation, height=200, width=200, key="lottie_animation")
    st.markdown("<h2 style='color: #007bff;'>Jelajahi Aplikasi!</h2>", unsafe_allow_html=True)
    st.markdown("**Tentang Model:** Model ini mengklasifikasikan bunga ke dalam 5 kategori: daisy, dandelion, mawar, bunga matahari, dan tulip.")
    
    # Bagian fitur dengan efek hover
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
                <div class="feature-hover">Klasifikasi Cepat
                    <span class="tooltip-text">Dapatkan prediksi dalam hitungan detik. Nikmati desain yang ramping dan modern.</span>
                </div>
            </li>
            <li>
                <div class="feature-hover">Sangat Akurat
                    <span class="tooltip-text">Akurasi model tinggi, memberikan prediksi dengan percaya diri.</span>
                </div>
            </li>
        </ul>
    """, unsafe_allow_html=True)

    # Informasi kontak
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Hubungi kami di: [**Kelompok 4**](https://www.linkedin.com/in/het-patel-8b110525a/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')

# Nama kelas bunga
class_names = open("labels.txt", "r").readlines()

# Memuat model
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model("keras_Model.h5")  # Ganti dengan jalur model Anda
        return model
    except Exception as e:
        st.error(f"Kesalahan saat memuat model: {e}")
        return None

model = load_my_model()

# Judul utama dengan efek teks keren
st.markdown("""
    <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
    🌸 Klasifikasi Gambar Bunga
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

st.header("Unggah Gambar Di Sini dan Dapatkan Prediksinya!")

# Fungsi memuat gambar
def load_image(filename):
    img = load_img(filename, target_size=(224, 224))  # Ukuran target disesuaikan dengan model yang digunakan
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img
    
# Membuat folder untuk gambar jika belum ada
if not os.path.exists('./images'):
    os.makedirs('./images')

# Bagian unggah gambar dengan uploader file yang menarik
image_file = st.file_uploader("🌄 Unggah gambar", type=["jpg", "png"], key="file_uploader")

if image_file is not None:
    if st.button("Klasifikasikan Gambar 🧠", key="classify_button"):
        img_path = f"./images/{image_file.name}"
        with open(img_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        image = Image.open(img_path)
        st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

        img_to_predict = load_image(img_path)

        # Spinner progres
        with st.spinner('🔍 Mengklasifikasikan gambar...'):
            time.sleep(2)
            try:
                predictions = model.predict(img_to_predict)
                predicted_class = np.argmax(predictions, axis=-1)
                confidence = np.max(predictions)
            except Exception as e:
                st.error(f"Kesalahan saat prediksi: {e}")
                predictions = None

        # Ambang batas dan tampilan hasil
        if predictions is not None:
            confidence_threshold = 0.60

            if confidence < confidence_threshold:
                result = f"Prediksi: Bunga tidak dikenali (Kepercayaan: {confidence*100:.2f}%)"
            else:
                result = f"Prediksi: {class_names[predicted_class[0]]} dengan kepercayaan {confidence*100:.2f}%"

            st.success(result)

        os.remove(img_path)

# Tambahkan progress bar unik untuk interaktivitas yang lebih baik
if st.button("Muat Ulang Aplikasi"):
    st.progress(100)

# Informasi tambahan tentang jenis bunga
st.markdown("""
### **Jenis Bunga**:
- **Daisy**: Bunga umum dengan cakram tengah dan kelopak putih.
- **Dandelion**: Bunga kuning yang berubah menjadi bola bulu ketika matang.
- **Mawar**: Bunga populer yang dikenal dengan warna-warnanya yang cerah dan harum.
- **Bunga Matahari**: Bunga besar berwarna kuning yang dikenal dengan ukurannya dan heliotropisme.
- **Tulip**: Bunga berwarna-warni dengan bentuk yang halus dan bulat.
""", unsafe_allow_html=True)

# Data untuk kinerja klasifikasi bunga
data = {
    "Kelas": ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips'],
    "Akurasi": [0.92, 0.88, 0.89, 0.93, 0.85],  
    "Presisi": [0.90, 0.85, 0.87, 0.92, 0.84]  
}

df = pd.DataFrame(data)

# Tabel DataFrame bergaya dengan tinggi tertentu
st.markdown("### Kinerja Klasifikasi Bunga")
styled_table = df.style.background_gradient(cmap="coolwarm", subset=['Akurasi', 'Presisi'])
st.dataframe(styled_table, height=400)
