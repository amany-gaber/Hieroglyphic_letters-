import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import os
import tempfile
import uuid

# Load YOLO Model
MODEL_PATH = "hieroglyphic_model.pt"
model = YOLO(MODEL_PATH)

st.title("Hieroglyph Detection App 🚀")
st.write("Upload an image to detect hieroglyphic symbols.")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Run YOLO inference
    results = model(image)
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    unique_classes = set()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
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

            unique_classes.add(class_name)

    st.image(annotated_image, caption="Processed Image", use_column_width=True)
    st.write(f"**Detected Classes:** {', '.join(unique_classes)}")

    # Save the image for download
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    annotated_image.save(temp_file.name, format="PNG")

    with open(temp_file.name, "rb") as file:
        st.download_button(label="Download Processed Image", data=file, file_name="annotated_image.png", mime="image/png")
