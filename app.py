import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set the title and  a brief description
st.title("Dog-Cat Image Classification App")
st.write("Upload an image of a cat or dog, and we'll predict which is it")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image..", type = ["jpg", "jpeg", "png"])

# Check if image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width="auto")
    st.write("")

    # Preprocess the image for prediction
    img = np.array(image)
    # resize the image
    img = tf.image.resize(img, (64, 64))
    # normalize the image
    img = img / 255.0
    img = np.expand_dims(img, axis = 0)

    # load the trained model
    model = load_model("vgg_model.h5")

    # Make preedictions
    prediction = model.predict(img)
    label = "Cat" if prediction[0][0] > 0.5 else "Dog"

    # Display the prediction
    st.write(f" ### Prediction: {label}")
