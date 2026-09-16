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

# Load Pre-trained YOLOv8 Base Model (Detects standard vehicles)
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

st.title("🚗 ParkVision AI: Intelligent Urban Parking Platform")
st.write("Upload a parking lot image for real-time occupancy detection, color-coded visual overlays, and intelligent driver guidance.")

# Sidebar Configuration
st.sidebar.header("Model Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)

# File Upload Section
uploaded_file = st.file_uploader("Upload Parking Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process Image Input using PIL
    image = Image.open(uploaded_file).convert("RGB")
    
    # Run YOLO Inference
    results = model.predict(source=np.array(image), conf=conf_threshold)
    boxes = results[0].boxes

    occupied_count = 0
    vehicle_classes = ["car", "truck", "bus", "motorcycle"]
    class_names = model.names
    
    # Prepare Drawing Canvas
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    # Process Detections
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label_name = class_names[cls_id].lower()

        # Count detected vehicles as occupied slots
        if label_name in vehicle_classes:
            occupied_count += 1
            box_color = "#FF0000"  # Red for occupied spaces
            
            # Draw Bounding Box & Label
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            draw.text((x1, max(y1 - 12, 0)), f"Occupied ({label_name.capitalize()})", fill=box_color)

    # Note: Estimating available capacity based on detected vehicles
    # You can adjust baseline_total_capacity to match your specific lot's capacity
    baseline_total_capacity = max(occupied_count + 5, 20)
    empty_count = max(0, baseline_total_capacity - occupied_count)
    occupancy_rate = (occupied_count / baseline_total_capacity * 100) if baseline_total_capacity > 0 else 0.0

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
        st.image(annotated_img, caption="Red Bounding Boxes: Detected Occupied Vehicles", use_container_width=True)

    with col2:
        st.subheader("Parking Utilization Analytics")
        m1, m2 = st.columns(2)
        m1.metric("Estimated Total Capacity", baseline_total_capacity)
        m2.metric("Estimated Available Slots", empty_count)

        m3, m4 = st.columns(2)
        m3.metric("Occupied Vehicles Detected", occupied_count)
        m4.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")

        st.markdown("---")
        st.subheader("Smart Guidance")
        st.markdown(f"**Congestion Level:** :{status_color}[{congestion_status}]")
        st.info(f"**Recommendation:** {recommendation}")
