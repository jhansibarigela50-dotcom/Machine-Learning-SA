import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Streamlit Page Setup
st.set_page_config(
    page_title="ParkVision AI - Urban Parking Analytics",
    page_icon="🚗",
    layout="wide"
)

# Load Model (Tries best.pt first, falls back to yolov8n.pt)
@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("best.pt"), "custom"
    except Exception:
        return YOLO("yolov8n.pt"), "coco"

model, model_type = load_yolo_model()

st.title("🚗 ParkVision AI: Intelligent Urban Parking Platform")
st.write("Upload a parking lot image for real-time occupancy detection, color-coded visual overlays, and intelligent driver guidance.")

# Sidebar Configuration
st.sidebar.header("Model Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.25, 0.05)

# Display model source status in sidebar
if model_type == "custom":
    st.sidebar.success("Loaded custom model: best.pt")
else:
    st.sidebar.warning("Using base YOLOv8 model (best.pt not detected)")

# File Upload Section
uploaded_file = st.file_uploader("Upload Parking Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Run YOLO Inference
    results = model.predict(source=np.array(image), conf=conf_threshold)
    boxes = results[0].boxes

    occupied_count = 0
    empty_count = 0
    class_names = model.names
    
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label_name = class_names[cls_id].lower()

        # Logic for custom trained dataset with "empty" / "occupied" classes
        if "empty" in label_name:
            empty_count += 1
            box_color = "#00FF00"  # Green for empty space
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            
        elif "occupied" in label_name:
            occupied_count += 1
            box_color = "#FF0000"  # Red for occupied space
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            
        # Fallback logic if using standard COCO vehicle classes (car, truck, bus)
        elif label_name in ["car", "truck", "bus", "motorcycle"]:
            occupied_count += 1
            box_color = "#FF0000"  # Red for detected vehicles
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

    total_slots = occupied_count + empty_count
    
    # Handle scenario where only vehicles were detected (COCO fallback)
    if total_slots == 0 and occupied_count > 0:
        total_slots = occupied_count

    occupancy_rate = (occupied_count / total_slots * 100) if total_slots > 0 else 0.0

    # Decision Engine Logic
    if occupancy_rate < 40:
        congestion_status = "Low Congestion"
        status_color = "green"
        recommendation = "Slots available — proceed to park."
    elif 40 <= occupancy_rate <= 75:
        congestion_status = "Moderate Congestion"
        status_color = "orange"
        recommendation = "Slots available, but space is filling up fast."
    else:
        congestion_status = "High Congestion"
        status_color = "red"
        recommendation = "Parking lot full — search for another location."

    # Render User Interface Layout
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Visual Overlay Output")
        st.image(annotated_img, caption="Green: Available Slot | Red: Occupied Slot", use_container_width=True)

    with col2:
        st.subheader("Parking Utilization Analytics")
        m1, m2 = st.columns(2)
        m1.metric("Total Slots Detected", total_slots)
        m2.metric("Available Slots", empty_count)

        m3, m4 = st.columns(2)
        m3.metric("Occupied Slots", occupied_count)
        m4.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")

        st.markdown("---")
        st.subheader("Smart Guidance")
        st.markdown(f"**Congestion Level:** :{status_color}[{congestion_status}]")
        st.info(f"**Recommendation:** {recommendation}")
