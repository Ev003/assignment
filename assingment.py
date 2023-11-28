import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_log(image):
    log_image = cv2.GaussianBlur(image, (3, 3), 0)
    log_image = cv2.Laplacian(log_image, cv2.CV_64F)
    log_image = np.uint8(np.absolute(log_image))
    return log_image

def apply_dog(image):
    blurred1 = cv2.GaussianBlur(image, (5, 5), 0)
    blurred2 = cv2.GaussianBlur(image, (9, 9), 0)
    dog_image = blurred1 - blurred2
    dog_image = np.uint8(np.absolute(dog_image))
    return dog_image

def apply_canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges

def process_image(image, crop_params, rotate_angle, grayscale):
    # Crop image
    if crop_params:
        x, y, w, h = crop_params
        image = image[y:y+h, x:x+w]

    # Rotate image
    if rotate_angle:
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, 1)
        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    # Convert to grayscale
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image

def main():
    st.title("Image Processing with Streamlit")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the input image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Image processing options
        crop_params = st.sidebar.text_input("Crop Parameters (x, y, width, height):")
        crop_params = [int(param) for param in crop_params.split(",")] if crop_params else None

        rotate_angle = st.sidebar.slider("Rotate Angle (degrees)", -180, 180, 0)

        grayscale = st.sidebar.checkbox("Convert to Grayscale")

        # Apply selected image processing options
        processed_image = process_image(image_np, crop_params, rotate_angle, grayscale)

        # Select algorithm
        algorithm = st.radio("Select Image Processing Algorithm", ("LOG", "DOG", "Canny"))

        # Apply selected algorithm
        if algorithm == "LOG":
            result = apply_log(processed_image)
        elif algorithm == "DOG":
            result = apply_dog(processed_image)
        else:
            result = apply_canny(processed_image)

        # Display original and processed images
        st.image(image, caption='Original Image', use_column_width=True)

        st.subheader(f"{algorithm} Result")

        # Display processed image with a specific colormap
        if algorithm == "LOG" or algorithm == "DOG":
            plt.imshow(result, cmap='gray')
            plt.axis('off')
            st.pyplot()
        else:
            st.image(result, caption=f'{algorithm} Result', use_column_width=True)

if __name__ == "__main__":
    main()
