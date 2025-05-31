import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")

# Sidebar - Drawing options
st.sidebar.title("Drawing Settings")
mode = st.sidebar.selectbox("Drawing Tool", ("freedraw", "line"))
stroke_width = st.sidebar.slider("Stroke width", 5, 25, 15)
# stroke_color = st.sidebar.color_picker("Stroke color", "#000000")
# bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")

# Load the trained model
@st.cache_resource
def load_mnist_model():
    return load_model("mnist_model.keras")

model = load_mnist_model()

# Preprocessing function
def preprocess(img):
    img_np = np.array(img)

    if img_np.shape[2] == 4: 
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    elif img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    # normalized = resized.astype("float32") #/ 255.0
    input_data = np.expand_dims(resized, axis=(0, -1))

    return input_data

# User input section

choice = st.selectbox("Choose Input Method", ("Drawing Tool", "Upload Image"))
col1, col2 = st.columns(2)

input_img = None

with col1:

    if choice == "Drawing Tool":
        st.subheader("🖌️ Draw a Digit")
        canvas_result = st_canvas(
            stroke_width=stroke_width,
            stroke_color= "#FFFFFF", #stroke_color,
            background_color="#000000", #bg_color,
            height=200,
            width=200,
            drawing_mode=mode,
            key="canvas"
        )
        if canvas_result.image_data is not None:
            input_img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    else:
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        if uploaded_file:
            input_img = Image.open(uploaded_file).convert("RGB")

with col2:
    st.subheader("Image")
    if input_img:
        st.image(input_img, caption="Input Image", width=200)

# Prediction
if st.button("🔍 Predict"):
    if input_img is not None:
        processed = preprocess(input_img)
        prediction = model.predict(processed)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        st.success(f"🧠 Predicted Digit: **{digit}** with **{confidence:.2f}%** confidence")
    else:
        st.warning("⚠️ Please draw or upload an image before predicting.")
