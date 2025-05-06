# To run: streamlit run app.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import requests
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile

# --- Constants ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=138cFx1o_9wOKGNRezJ2GVGhefOStjZcp"
MODEL_PATH = "baseline_cnn.h5"

# --- Download model if not present ---
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            r = requests.get(MODEL_URL, allow_redirects=True)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        st.success("Model downloaded.")
    else:
        st.info("Model already cached.")

# --- Preprocessing function ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Shape: (1, 224, 224, 1)
    return img

# --- Prediction function ---
def predict(model, image_path):
    st.info("🧠 Machine Analyzing...")
    img = preprocess_image(image_path)
    pred_prob = model.predict(img)[0][0]
    pred_class = "Pneumonia" if pred_prob > 0.5 else "Normal"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
    return pred_class, confidence

# --- Load model (cached) ---
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

# --- Streamlit UI ---
st.markdown("<div align='center'><h1 style='font-size: 48px;'>Sehatman Pakistan</h1></div>", unsafe_allow_html=True)
st.markdown("<div align='center'><p style='font-size: 28px;'>X-ray Pneumonia Detection</p></div>", unsafe_allow_html=True)

# Ensure model is available
download_model()
model = load_model(MODEL_PATH)

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_filename = tmp_file.name

    # Prediction
    pred_class, confidence = predict(model, temp_filename)

    # Display result
    if pred_class == "Pneumonia":
        st.error(f"🩺 Prediction: **{pred_class}**")
    else:
        st.success(f"🩺 Prediction: **{pred_class}**")

    st.info(f"Confidence Score: `{confidence:.1%}`")

    # Display uploaded image
    image = Image.open(temp_filename)
    st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
