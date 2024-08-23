import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the model
MODEL_PATH = "eye_model.keras"
model = load_model(MODEL_PATH)

# Define the disease labels based on your model's training
disease_labels = [
    'Age-Related Macular Degeneration', 'Branch Retinal Vein Occlusion', 'Cataract',
    'Diabetic Retinopathy', 'Drusen', 'Glaucoma', 'Hypertension',
    'Media Haze', 'Normal', 'Others', 'Pathological Myopia', 'Tessellation'
]

# Streamlit app interface
st.title("Multiple Eye Disease Classification")
st.write("Upload an eye image to classify the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', width=800)

    # Preprocess the image to match your model's input
    img = img.resize((224, 224))  # Adjust to your model's input size
    img = img_to_array(img) / 255.0  # Convert image to array and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img)
    predicted_class = disease_labels[np.argmax(prediction)]

    # Display the prediction
    st.markdown(
        f"<h1>Prediction: <span style='color:red;'>{predicted_class}</span> <span style='color:yellow;'>Disease</span>",
        unsafe_allow_html=True
    )