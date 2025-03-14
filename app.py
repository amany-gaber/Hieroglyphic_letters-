import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import os
from ultralytics import YOLO
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

class YOLOModelHandler:
    """Enhanced YOLO Model Handler for Hieroglyph Detection."""
    
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load YOLO model with improved error handling and logging."""
        try:
            model = YOLO(model_path)
            st.success(f"✅ Model loaded successfully")
            return model
        except Exception as e:
            st.error(f"❌ Error loading YOLO model: {e}")
            return None
    
    def run_inference(self, image):
        """Advanced inference with additional metadata and robust error handling."""
        if self.model is None:
            return [], image
        
        results = self.model(image)
        
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        try:
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            detections = []
            unique_classes = set()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Enhanced visualization
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    
                    label = f"{class_name} {conf:.2f}"
                    draw.text(
                        (x1, max(0, y1-25)), 
                        label, 
                        fill="white", 
                        font=font, 
                        stroke_width=3, 
                        stroke_fill="black"
                    )
                    
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "box": [x1, y1, x2, y2]
                    })
                    unique_classes.add(class_name)
            
            return list(unique_classes), annotated_image
        except Exception as e:
            st.error(f"Error during inference: {e}")
            return [], image

# Initialize model handler
@st.cache_resource
def load_model():
    return YOLOModelHandler("hieroglyphic_model.pt")

model_handler = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Process image
    with st.spinner("Detecting hieroglyphs..."):
        unique_classes, annotated_image = model_handler.run_inference(image)
    
    # Display results
    with col2:
        st.subheader("Detected Hieroglyphs")
        st.image(annotated_image, use_column_width=True)
    
    # Display detected classes
    if unique_classes:
        st.success(f"Detected {len(unique_classes)} unique hieroglyphic symbols")
        st.write("Detected symbols:")
        for i, class_name in enumerate(sorted(unique_classes)):
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
else:
    st.info("Please upload an image to detect hieroglyphs")

# Add footer
st.markdown("---")
st.markdown("Hieroglyph Detection App using YOLO - Created by Amany Gaber")
