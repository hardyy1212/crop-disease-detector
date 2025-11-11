import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('crop_disease_model.h5')
        return model
    except Exception as e:
        st.warning("⚠️ Model not found or failed to load. Please upload a valid 'crop_disease_model.h5' file.")
        st.error(f"Error: {e}")
        return None

model = load_model()

# ----------------------------
# Class names and Solutions
# ----------------------------
class_names = [
    'Healthy',
    'Tomato_Bacterial_spot',
    'Potato_Late_blight',
    'Tomato_Leaf_Mold'
]

solutions = {
    'Healthy': '✅ No action needed. Your crop looks good!',
    'Tomato_Bacterial_spot': '🧴 Apply copper-based fungicide and improve air circulation. Avoid overhead watering.',
    'Potato_Late_blight': '🌿 Use fungicides like chlorothalonil. Remove and destroy infected plants immediately.',
    'Tomato_Leaf_Mold': '🍃 Increase ventilation and apply fungicide. Avoid wetting leaves.'
}

# ----------------------------
# Preprocessing function
# ----------------------------
def preprocess_image(image):
    try:
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Crop Disease Detector 🌱", layout="centered")
st.title("🌾 AI Crop Disease Detector")
st.write("Upload a **leaf image** of your crop to detect potential diseases and get treatment suggestions.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Disease"):
        if model is None:
            st.error("Model not loaded. Please ensure 'crop_disease_model.h5' is in the app directory.")
        else:
            processed_image = preprocess_image(image)
            if processed_image is not None:
                predictions = model.predict(processed_image)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100

                st.success(f"**Detected Disease:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.info(f"**Suggested Solution:** {solutions.get(predicted_class, 'Consult a local agricultural expert.')}")

                # Optional: Show all class probabilities
                with st.expander("📊 Prediction Details"):
                    for i, prob in enumerate(predictions[0]):
                        st.write(f"{class_names[i]}: {prob*100:.2f}%")
