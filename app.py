import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import io # Import the io module

# --- Load the trained model ---
# Make sure 'best_digit_model.keras' is in the same directory as app.py,

MODEL_PATH = 'best_digit_model.keras'

@st.cache_resource 
def load_my_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model(MODEL_PATH)

# --- Define the preprocessing function ---

def preprocess_image_simple_resize(image_bytes, invert_colors=False):
    """
    Loads, preprocesses, and resizes an image to 28x28 directly
    from image bytes (from st.file_uploader).

    Args:
        image_bytes (bytes): The image content as bytes from st.file_uploader.
        invert_colors (bool): Set to True if the input image has light digits
                              on a dark background. Defaults to False (dark digits
                              on a light background, like MNIST).

    Returns:
        np.ndarray: Preprocessed image array (1, 28, 28, 1) or None if error.
    """
    try:
        # Convert bytes to a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Read image in grayscale using cv2.imdecode
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Could not decode image.")

        # invert if digit is black on white (MNIST is black on white)
        if invert_colors:

            if img.dtype != np.uint8:
                 img = img.astype(np.uint8)
            img = cv2.bitwise_not(img)

        # Resize the entire image to 28x28
        resized_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        # Apply a simple threshold to ensure black/white
        _, final_processed_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Normalize to [0, 1]
        final_processed_img = final_processed_img.astype(np.float32) / 255.0

        # Reshape for model input: (1, 28, 28, 1)
        final_processed_img = final_processed_img.reshape(1, 28, 28, 1)

        return final_processed_img

    except Exception as e:

        print(f"An error occurred during image preprocessing: {e}") 
        return None

# --- Streamlit App Layout ---
st.title("Handwritten Digit Recognition")

st.write("Upload an image of a handwritten digit (0-9) To predict it.")

st.write("iNSTRUCTIONS: Write your digit in paint with a large brush and crop it to a square shape with a bit of padding.")

st.write("Check the github for sample screenshots and documentatio - '/Moznaim/digit-detection'")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # To display the image
    image = uploaded_file.getvalue()
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if model is not None:
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        # Adjust invert_colors based on the expected format of uploaded images
        # If images are dark digits on light background (like MNIST), set invert_colors=False
        # If images are light digits on dark background, set invert_colors=True
        preprocessed_image = preprocess_image_simple_resize(image, invert_colors=True)

        if preprocessed_image is not None:
            # Make a prediction
            predictions = model.predict(preprocessed_image)
            predicted_label = np.argmax(predictions)
            confidence = predictions[0][predicted_label] * 100

            st.success(f"Prediction: The digit is **{predicted_label}** with **{confidence:.2f}%** confidence.")

        else:
            st.error("Image preprocessing failed.")
    else:
        st.error("Model not loaded. Cannot make a prediction.")