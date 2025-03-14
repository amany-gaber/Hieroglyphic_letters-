import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load the model (cached to avoid reloading on each run)
@st.cache_resource
def load_model():
    model = torch.load("Keywords (1) (1).pt", map_location=torch.device('cpu'))
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Define Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),          # Convert to tensor
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit UI
st.title("Hieroglyphic Letter Classifier 🏺")
st.write("Upload an image of a Hieroglyphic letter to classify it.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    input_tensor = preprocess_image(image)

    # Make Prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    st.write(f"🔍 **Predicted Hieroglyphic Letter:** {predicted_class}")
