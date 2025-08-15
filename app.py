# app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --- Configuration & Model Loading ---
# NOTE: The model file (e.g., 'best_model.pth') must be in the same directory as this script.
MODEL_PATH = 'best_model.pth'
NUM_CLASSES = 6
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMAGE_SIZE = (224, 224)

# Define the model architecture. This must match the model used for training.
# We'll use the same function from the training script to ensure consistency.
def build_model(model_name, num_classes):
    """
    Loads a pre-trained model and adds a custom classifier.
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=None) # No weights needed, we'll load them later
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Unsupported model name.")
    return model

# Load the trained model.
# NOTE: Replace 'mobilenet_v2' with the model you actually trained.
@st.cache_resource
def load_trained_model():
    """
    Loads the model state from the saved .pth file.
    This function is cached by Streamlit to avoid reloading the model on every interaction.
    """
    try:
        # Build the architecture first
        model = build_model('mobilenet_v2', NUM_CLASSES)
        
        # Load the state dictionary
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        
        # Set the model to evaluation mode
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found. Please ensure '{MODEL_PATH}' is in the same directory.")
        return None

# Get the model instance
model = load_trained_model()

# Define the image transforms for inference
preprocess = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Prediction Function ---
def predict_image_class(image):
    """
    Makes a prediction on the given image using the loaded model.
    """
    if model is None:
        return None, "Model not loaded."
    
    # Preprocess the image
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0) # Add a batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
    
    return predicted_class, None

# --- Streamlit UI ---
st.set_page_config(
    page_title="Trash Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Trash Classifier")
st.markdown("Upload an image of a piece of trash, and I'll tell you what category it belongs to!")

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        
        with st.spinner("Analyzing..."):
            predicted_class, error = predict_image_class(image)
            if error:
                st.error(error)
            else:
                st.success(f"Prediction: **{predicted_class.capitalize()}**")

else:
    st.info("The model could not be loaded. Please check the console for details.")

st.markdown("---")
st.markdown("This application uses a pre-trained image classification model fine-tuned on the TrashNet dataset.")
