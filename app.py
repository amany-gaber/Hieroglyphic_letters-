import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import os
import numpy as np
import torch
import tempfile

# Set page config
st.set_page_config(
    page_title="Hieroglyph Detection App",
    page_icon="🏺",
    layout="wide"
)

# App title and description
st.title("Hieroglyph Detection App")
st.markdown("Upload an image to detect hieroglyphic symbols")

# Check if model file exists
if not os.path.exists("hieroglyphic_model.pt"):
    st.error("Model file not found. Please make sure 'hieroglyphic_model.pt' is in the app directory.")
    st.stop()

@st.cache_resource
def load_model():
    """Load the YOLO model with better error handling."""
    try:
        import ultralytics
        from ultralytics import YOLO
        
        st.info(f"Using ultralytics version: {ultralytics.__version__}")
        st.info(f"Using torch version: {torch.__version__}")
        
        model = YOLO("hieroglyphic_model.pt")
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {str(e)}")
        st.info("Detailed error information for debugging:")
        st.code(str(e))
        return None

# Load the model
model = load_model()

def process_image(image, model):
    """Process the image with the YOLO model."""
    if model is None:
        return [], image
    
    try:
        # Convert PIL Image to numpy array for YOLO
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array)
        
        # Create a copy of the image for drawing
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        unique_classes = set()
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                draw.text(
                    (x1, max(0, y1-25)), 
                    label, 
                    fill="white", 
                    font=font, 
                    stroke_width=3, 
                    stroke_fill="black"
                )
                
                unique_classes.add(class_name)
        
        return list(unique_classes), annotated_image
    
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        st.info("Detailed error information for debugging:")
        st.code(str(e))
        return [], image

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    try:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process image
        with st.spinner("Detecting hieroglyphs..."):
            unique_classes, annotated_image = process_image(image, model)
        
        # Display results
        with col2:
            st.subheader("Detected Hieroglyphs")
            st.image(annotated_image, use_column_width=True)
        
        # Display detected classes
        if unique_classes:
            st.success(f"Detected {len(unique_classes)} unique hieroglyphic symbols")
            st.write("Detected symbols:")
            for class_name in sorted(unique_classes):
                st.write(f"- {class_name}")
        else:
            st.info("No hieroglyphs detected in the image")
        
        # Download button for annotated image
        buf = io.BytesIO()
        annotated_image.save(buf, format="PNG")
        btn = st.download_button(
            label="Download Annotated Image",
            data=buf.getvalue(),
            file_name="annotated_hieroglyphs.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")
else:
    st.info("Please upload an image to detect hieroglyphs")

# Add footer
st.markdown("---")
st.markdown("Hieroglyph Detection App using YOLO - Created by Amany Gaber")
