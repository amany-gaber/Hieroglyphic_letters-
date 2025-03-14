import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model function
@st.cache_resource
def load_model():
    model = torch.load("hieroglyphic_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

# Load the model
model = load_model()

# Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Streamlit UI
st.title("🔠 Hieroglyphic Letter Classifier")
st.write("📷 Upload a Hieroglyphic letter image for classification.")

uploaded_file = st.file_uploader("📥 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Uploaded Image", use_column_width=True)

    # Process image
    input_tensor = preprocess_image(image)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    st.write(f"📝 **Predicted Hieroglyphic Letter:** {predicted_class}")
