import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import io
import tempfile
import uuid

# ✅ Initialize YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO("hieroglyphic_model.pt")  # Update with your actual path
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ✅ Image Upload Section
st.title("🔠 Hieroglyphic Letter Detector")
st.write("Upload an image containing Hieroglyphic symbols to detect them.")

uploaded_file = st.file_uploader("📥 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 Uploaded Image", use_column_width=True)

    # ✅ Run YOLO Detection
    if model:
        with st.spinner("Detecting symbols... ⏳"):
            results = model(image)

            # Create a copy of the image to draw on
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

                    # Draw bounding box and label
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    label = f"{class_name} {conf:.2f}"
                    draw.text((x1, max(0, y1 - 25)), label, fill="white", font=font, stroke_width=3, stroke_fill="black")

                    unique_classes.add(class_name)

            st.image(annotated_image, caption="📌 Detected Symbols", use_column_width=True)

            # ✅ Show detected classes
            if unique_classes:
                st.write("### 📝 Detected Symbols:")
                for symbol in unique_classes:
                    st.write(f"- **{symbol}**")
            else:
                st.warning("⚠️ No symbols detected.")

            # ✅ Allow user to download the annotated image
            output_filename = f"{uuid.uuid4()}_annotated.png"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            annotated_image.save(output_path, format="PNG")

            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download Annotated Image",
                    data=file,
                    file_name="annotated_hieroglyphic.png",
                    mime="image/png"
                )
    else:
        st.error("❌ Model not loaded. Please check the model path.")

# ✅ Add Health Check Section
st.sidebar.header("ℹ️ Model Health Check")
if model:
    st.sidebar.success("✅ Model is loaded and ready!")
else:
    st.sidebar.error("❌ Model failed to load.")
