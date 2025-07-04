import streamlit as st
import cv2
import PIL.Image
import numpy as np

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded file into an image using PIL
    image = PIL.Image.open(uploaded_file)

    # Convert the image to a numpy array for OpenCV processing
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Load Haar cascade for face detection
    cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_file)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the image back to RGB for display in Streamlit
    result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Display the processed image in Streamlit
    st.image(result_image, caption="Processed Image with Face Detection", use_column_width=True)
