import os
import h5py
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.abspath(os.path.join(working_dir, '..', 'trained_models', 'fileh.h5'))
print("File exists:", os.path.exists(model_path))
class_indices_path = os.path.abspath(os.path.join(working_dir, '..', 'trained_models', 'class_indices.json'))
try:
    with h5py.File(model_path, 'r') as f:
        print("HDF5 file opened successfully.")
except Exception as e:
    print("Failed to open HDF5 file:", e)
    st.error(f"Failed to open model file: {e}")
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  


model = tf.keras.models.load_model(model_path)

try:
    with open(class_indices_path) as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"Error loading class indices json: {e}")
    st.stop()



index_to_class = {v: k for k, v in class_indices.items()}


def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image_file, index_to_class):
    try:
        preprocessed_img = load_and_preprocess_image(image_file)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = index_to_class[predicted_class_index]
        return predicted_class_name
    except Exception as e:
        return f"Prediction error: {e}"



st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, index_to_class)
            st.success(f'Prediction: {prediction}')
