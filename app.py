import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define the model architecture (must match training)
class HieroglyphicModel(nn.Module):
    def __init__(self):
        super(HieroglyphicModel, self).__init__()
        # Define model layers (Modify based on your actual model)
        self.layer1 = nn.Linear(512, 128)
        self.layer2 = nn.Linear(128, 10)  # Adjust for the number of classes

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Load model function (cached to prevent reloads)
@st.cache_resource
def load_model():
    model = HieroglyphicModel()
    model.load_state_dict(torch.load("hieroglyphic_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

# Load the model
model = load_model()

# Define Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),          # Convert to tensor
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# UI Title
st.title("Hieroglyphic Letter Classifier 🏺")
st.write("Upload an image of a Hieroglyphic letter to classify it.")

# File Uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    input_tensor = preprocess_image(image)

    # Make Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert to probabilities
        top3_probs, top3_indices = torch.topk(probabilities, 3)  # Get top 3 predictions

    # Display predictions
    st.write("🔍 **Predicted Classes & Confidence:**")
    for i in range(3):  # Show top 3 predictions
        st.write(f"📖 Class: {top3_indices[0][i].item()}, Confidence: {top3_probs[0][i].item()*100:.2f}%")
