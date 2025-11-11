import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('crop_disease_model.h5')
    except Exception as e:
        st.error("❌ Model not found. Please ensure 'crop_disease_model.h5' is in the same folder.")
        st.error(str(e))
        return None
    return model

model = load_model()

class_names = ['Healthy', 'Tomato_Bacterial_spot', 'Potato_Late_blight', 'Tomato_Leaf_Mold']
solutions = {
    'Healthy': '✅ No action needed. Your crop looks good!',
    'Tomato_Bacterial_spot': '🧴 Apply copper-based fungicide and improve air circulation. Avoid overhead watering.',
    'Potato_Late_blight': '🪓 Remove infected leaves and apply chlorothalonil-based fungicide.',
    'Tomato_Leaf_Mold': '🌿 Increase ventilation and apply fungicide. Avoid wetting leaves.'
}

def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

st.set_page_config(page_title="AI Crop Disease Detector", page_icon="🌱", layout="wide")
st.title("🌾 AI Crop Disease Detector")
st.write("Upload a leaf image to detect diseases and get expert solutions instantly.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Disease"):
        if model is None:
            st.error("Model not loaded. Please check the model file.")
        else:
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100

            st.success(f"**Detected Disease:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.info(f"**Suggested Solution:** {solutions.get(predicted_class, 'Consult an agricultural expert.')}")

            st.subheader("Prediction Details")
            for i, prob in enumerate(predictions[0]):
                st.write(f"{class_names[i]}: {prob * 100:.2f}%")

    if st.button("🔁 Reset"):
        st.experimental_rerun()
