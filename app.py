import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("best_model.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Streamlit App Title
st.title("🌿 Plant Disease Detection Web App")
st.markdown("Upload an image of a plant leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show prediction
    st.success(f"🌿 Predicted Disease: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")

else:
    st.warning("Please upload a plant leaf image to proceed.")
