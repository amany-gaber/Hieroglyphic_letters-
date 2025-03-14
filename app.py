import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# ✅ تحميل المودل مباشرة لأنه محفوظ كـ .pt
@st.cache_resource
def load_model():
    model = torch.load("hieroglyphic_model.pt", map_location=torch.device("cpu"))
    model.eval()  # وضع التقييم
    return model

# تحميل المودل
model = load_model()

# ✅ دالة تجهيز الصورة قبل إدخالها للمودل
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # إضافة batch dimension

# ✅ واجهة Streamlit
st.title("🔠 Hieroglyphic Letter Classifier")
st.write("📷 قم برفع صورة تحتوي على حرف هيروغليفي وسيتم تصنيفه.")

uploaded_file = st.file_uploader("📥 ارفع صورة", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📌 الصورة التي تم تحميلها", use_column_width=True)

    # تجهيز الصورة
    input_tensor = preprocess_image(image)

    # 🔍 تشغيل المودل للحصول على التوقعات
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    # عرض النتيجة
    st.write(f"📝 **الحرف الهيروغليفي المتوقع:** {predicted_class}")
