import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Streamlit Page Setup
st.set_page_config(
    page_title="ParkVision AI - Urban Parking Analytics",
    page_icon="🚗",
    layout="wide"
)

# Load Trained Weights
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error("Model file 'best.pt' not found in project root. Place your trained model file here.")

st.title("🚗 ParkVision AI: Intelligent Urban Parking Platform")
st.write("Upload a parking lot image for real-time occupancy detection, color-coded visual overlays, and intelligent driver guidance.")

# Sidebar Configuration
st.sidebar.header("Model Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.35, 0.05)

# File Upload Section
uploaded_file = st.file_uploader("Upload Parking Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process Image Input
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Run Inference
    results = model.predict(source=img_array, conf=conf_threshold)
    boxes = results[0].boxes

    occupied_count = 0
    empty_count = 0
    class_names = model.names
    annotated_img = img_array.copy()

    # Draw Overlay Visuals
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label_name = class_names[cls_id].lower()

        if "empty" in label_name:
            empty_count += 1
            box_color = (0, 255, 0)  # Green box for empty spaces
        else:
            occupied_count += 1
            box_color = (0, 0, 255)  # Red box for occupied spaces

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            annotated_img,
            label_name.capitalize(),
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2
        )

    # Compute Occupancy Statistics
    total_slots = occupied_count + empty_count
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
        st.image(annotated_img, caption="Green: Empty | Red: Occupied", use_column_width=True)

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
