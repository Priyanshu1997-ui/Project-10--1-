import streamlit as st
import torch
from PIL import Image
import tempfile
import os

# ==========================
# 🧠 Load YOLOv3 Custom Model
# ==========================
@st.cache_resource
def load_model():
    cache_dir = os.path.join(tempfile.gettempdir(), f"yolov3_cache_{os.getpid()}")
    torch.hub.set_dir(cache_dir)
    model = torch.hub.load('ultralytics/yolov3', 'custom', path='models/yolov3.pt', force_reload=True)
    model.conf = 0.1  # lower confidence threshold for better recall
    return model

model = load_model()
st.success("✅ YOLOv3 OCR Model Loaded Successfully!")

# ==========================
# 🎨 Streamlit UI
# ==========================
st.title("🧾 Custom OCR Field Detector (YOLOv3)")
st.write("Upload an image to detect **fields**, **values**, and **units** from documents!")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

# ==========================
# 📸 Image Upload & Detection
# ==========================
if uploaded_file:
    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        image_path = os.path.abspath(tmp.name)

    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    st.write("🔍 Detecting fields... Please wait...")

    # Resize before inference
    resized_image_path = os.path.join(tempfile.gettempdir(), "resized.jpg")
    image.resize((640, 640)).save(resized_image_path)

    # Run inference
    results = model(resized_image_path, size=640)

    # ==========================
    # 💾 Save & Display Results
    # ==========================
    output_dir = "runs/detect"
    os.makedirs(output_dir, exist_ok=True)
    results.save(save_dir=output_dir)

    df = results.pandas().xyxy[0]  # detection results dataframe

    if df.empty:
        st.warning("⚠️ No fields detected! Try lowering confidence or use another test image.")
    else:
        # Render detections on image
        results.render()
        detected_img = Image.fromarray(results.ims[0])
        st.image(detected_img, caption="✅ Detected Fields", use_container_width=True)

        st.write("📊 **Detection Results:**")
        st.dataframe(df)

else:
    st.info("Please upload an image to begin detection.")
