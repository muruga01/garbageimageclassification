import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Garbage Classification App", page_icon="🗑️")

# Title and description
st.title("🗑️ Garbage Classification App")
st.write("Upload an image of waste to classify it into one of the following categories: plastic, metal, glass, cardboard or paper.")

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('trashnet_efficientnetb0.h5')
    return model

model = load_model()

# Define class names (must match the order used during training)
class_names = ['plastic', 'metal', 'glass', 'cardboard', 'paper']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to 224x224
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    if img_array.shape[-1] != 3:  # Ensure RGB channels
        img_array = np.stack((img_array,) * 3, axis=-1) if len(img_array.shape) == 2 else img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess and predict
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    confidence_scores = predictions[0]
    
    # Get top prediction
    top_pred_idx = np.argmax(confidence_scores)
    top_pred_class = class_names[top_pred_idx]
    top_confidence = confidence_scores[top_pred_idx] * 100

    # Display top prediction
    st.subheader("Prediction")
    st.write(f"**Category**: {top_pred_class}")
    st.write(f"**Confidence**: {top_confidence:.2f}%")

    # Get top-3 predictions
    top_3_indices = np.argsort(confidence_scores)[-3:][::-1]  # Top 3 in descending order
    top_3_classes = [class_names[idx] for idx in top_3_indices]
    top_3_confidences = [confidence_scores[idx] * 100 for idx in top_3_indices]

    # Create a DataFrame for top-3 predictions
    top_3_df = pd.DataFrame({
        'Category': top_3_classes,
        'Confidence (%)': [f"{conf:.2f}" for conf in top_3_confidences]
    })

    # Display top-3 predictions
    st.subheader("Top-3 Predictions")
    st.table(top_3_df)

# Footer
st.markdown("---")
st.write("Built with Streamlit and TensorFlow | Dataset: TrashNet")