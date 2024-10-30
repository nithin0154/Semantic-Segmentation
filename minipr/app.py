import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from keras.utils import normalize
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt

# Define constants
SIZE = 256

# Load the trained model
model = load_model('model.h5')

# Streamlit title and file uploader
st.title('Image Segmentation Using U-Net')
st.subheader('Upload an image for segmentation')

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((SIZE, SIZE))
    image_array = np.array(image)
    
    # Display the input image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for prediction
    image_norm = np.expand_dims(normalize(image_array, axis=1), 2)
    image_norm = image_norm[:, :, 0][:, :, None]
    image_input = np.expand_dims(image_norm, 0)

    # Predict segmentation
    prediction = (model.predict(image_input)[0, :, :, 0] > 0.2).astype(np.uint8)

    # Save and display the prediction result
    prediction_image = Image.fromarray(prediction * 255)  # Convert 0/1 to 0/255 for better display
    st.image(prediction_image, caption='Segmented Image', use_column_width=True)

    # Save input and output images
    input_image_path = 'input.png'
    output_image_path = 'output.png'
    
    image.save(input_image_path)
    prediction_image.save(output_image_path)

    # Option to download images
    with open(input_image_path, "rb") as file:
        btn = st.download_button(
            label="Download input image",
            data=file,
            file_name="input.png",
            mime="image/png"
        )

    with open(output_image_path, "rb") as file:
        btn = st.download_button(
            label="Download output image",
            data=file,
            file_name="output.png",
            mime="image/png"
        )

# Option to close the app
st.button("Exit", on_click=lambda: st.stop())
