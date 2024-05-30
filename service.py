import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('segnet_model.keras')

uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Загруженное изображение', use_column_width=True)
    image_array = np.array(image.resize((128, 128))) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    st.write("Результат предсказания:", prediction)