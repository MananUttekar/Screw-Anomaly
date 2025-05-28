import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.markdown("""
<div style="background-color: #007BFF; padding: 19px; border-radius: 19px">
    <h2 style="color: white; text-align: left;">Screw Anomaly Detector</h2>
    Boost Your Quality Control with Screw Anomaly Detector - The Ultimate AI-Powered Inspection App. Try clicking a product image and watch how an AI Model will classify it between Good / Anomaly.

</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")


st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            color: #000000;  
        }
       section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div {
            color: #000000 ;
        }
""", unsafe_allow_html=True)

with st.sidebar:
    img = Image.open("./docs/screw-cartoon.png")
    st.image(img)
    st.subheader("About Screw Anomaly Detector")
    st.write(
        "Screw Anomaly Detector is a AI-powered application designed to help businesses "
        "streamline their quality control inspections..."
    )
    st.subheader("📂 Class Labels")
    st.markdown("- ✅ Good\n- ❌ Anomaly (scratched, bent head, etc.)")
    st.markdown("---")

# Function to load images
def load_uploaded_image(file):
    return Image.open(file).convert("RGB")

# Sidebar input method
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")
st.write("")
st.write("")


uploaded_file_img = None
camera_file_img = None

if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")

# Load Keras model
@st.cache_resource
def load_keras_model():
    model_path = r"C:\Users\VICTUS\Downloads\IntelAI\Week 13 - Dependencies\Dependencies\InspectorsAlly - Anomaly Detection\keras_model.h5"
    model = load_model(model_path)
    return model

model = load_keras_model()

# Anomaly Detection function using Keras model
def Anomaly_Detection(image: Image.Image):
    img_resized = image.resize((224, 224))
    img_array = (np.array(img_resized) / 127.5) - 1  # or normalize your way
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    preds = model.predict(img_array)[0]  # preds might be 1D or scalar

    # Fix for predicted class:
    if preds.ndim > 0:  # preds is array-like
        predicted_class = np.argmax(preds)
        confidence = preds[predicted_class]
    else:
        # scalar probability (binary classification)
        predicted_class = int(preds > 0.5)
        confidence = preds if predicted_class == 1 else 1 - preds

    if predicted_class == 0:
        return f"✅ Congratulations! Screw has been classified as a 'Good' item with {confidence:.2f} confidence."
    else:
        return f"⚠️ Anomaly Detected! Our system detected a defect with {confidence:.2f} confidence."

# Submission
submit = st.button(label="Submit a Screw Image")
if submit:
    st.subheader("Output")
    image_to_check = uploaded_file_img if input_method == "File Uploader" else camera_file_img

    if image_to_check:
        with st.spinner(text="This may take a moment..."):
            result = Anomaly_Detection(image_to_check)
            st.success(result)
    else:
        st.error("No image provided!")
