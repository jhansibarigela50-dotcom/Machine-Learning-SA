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

# Load Pre-trained YOLOv8 Base Model
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

st.title("🚗 ParkVision AI: Intelligent Urban Parking Platform")
st.write("Upload a parking lot image for real-time occupancy detection, color-coded visual overlays, and intelligent driver guidance.")

# Sidebar Configuration
st.sidebar.header("Model Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
grid_rows = st.sidebar.slider("Parking Grid Rows", 1, 10, 4)
grid_cols = st.sidebar.slider("Parking Grid Columns", 1, 15, 6)

# File Upload Section
uploaded_file = st.file_uploader("Upload Parking Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = image.size
    
    # Run YOLO Inference for Vehicles
    results = model.predict(source=np.array(image), conf=conf_threshold)
    boxes = results[0].boxes

    vehicle_classes = ["car", "truck", "bus", "motorcycle"]
    class_names = model.names
    
    # Extract vehicle bounding boxes
    detected_vehicles = []
    for box in boxes:
        cls_id = int(box.cls[0])
        label_name = class_names[cls_id].lower()
        if label_name in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_vehicles.append((x1, y1, x2, y2))

    # Helper function to calculate Intersection over Union (IoU) / Overlap
    def is_slot_occupied(slot_box, vehicle_boxes):
        sx1, sy1, sx2, sy2 = slot_box
        slot_area = (sx2 - sx1) * (sy2 - sy1)
        
        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            # Calculate overlap area
            ix1 = max(sx1, vx1)
            iy1 = max(sy1, vy1)
            ix2 = min(sx2, vx2)
            iy2 = min(sy2, vy2)
            
            if ix1 < ix2 and iy1 < iy2:
                overlap_area = (ix2 - ix1) * (iy2 - iy1)
                if (overlap_area / slot_area) > 0.20:  # 20% overlap threshold
                    return True
        return False

    # Generate Parking Space Grid Layout
    cell_w = img_w // grid_cols
    cell_h = img_h // grid_rows
    
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    occupied_count = 0
    empty_count = 0
    total_slots = grid_rows * grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            # Calculate coordinates for each slot
            sx1 = c * cell_w + 5
            sy1 = r * cell_h + 5
            sx2 = (c + 1) * cell_w - 5
            sy2 = (r + 1) * cell_h - 5
            
            slot_box = (sx1, sy1, sx2, sy2)
            
            if is_slot_occupied(slot_box, detected_vehicles):
                occupied_count += 1
                box_color = "#FF0000"  # Red for Occupied
                label = "Occupied"
            else:
                empty_count += 1
                box_color = "#00FF00"  # Green for Available
                label = "Available"

            # Draw Slot Rectangle
            draw.rectangle([sx1, sy1, sx2, sy2], outline=box_color, width=3)
            draw.text((sx1 + 5, sy1 + 5), label, fill=box_color)

    # Analytics Calculation
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

    # Render Interface
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Visual Overlay Output")
        st.image(annotated_img, caption="Green: Available Slot | Red: Occupied Slot", use_container_width=True)

    with col2:
        st.subheader("Parking Utilization Analytics")
        m1, m2 = st.columns(2)
        m1.metric("Total Defined Slots", total_slots)
        m2.metric("Available Slots", empty_count)

        m3, m4 = st.columns(2)
        m3.metric("Occupied Slots", occupied_count)
        m4.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")

        st.markdown("---")
        st.subheader("Smart Guidance")
        st.markdown(f"**Congestion Level:** :{status_color}[{congestion_status}]")
        st.info(f"**Recommendation:** {recommendation}")
