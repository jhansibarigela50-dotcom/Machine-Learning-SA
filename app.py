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

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

model = load_yolo_model()

st.title("🚗 ParkVision AI: Intelligent Urban Parking Platform")
st.write("Upload a parking lot image for real-time occupancy detection, color-coded visual overlays, and intelligent driver guidance.")

# Sidebar Configuration
st.sidebar.header("Model Controls")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.15, 0.05)
iou_threshold = st.sidebar.slider("Overlap Sensitivity", 0.05, 0.50, 0.15, 0.05)

# File Upload Section
uploaded_file = st.file_uploader("Upload Parking Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = image.size

    # Run YOLO Inference for vehicles
    results = model.predict(source=np.array(image), conf=conf_threshold)
    boxes = results[0].boxes

    vehicle_boxes = []
    vehicle_classes = [2, 3, 5, 7]  # COCO IDs for car, motorcycle, bus, truck
    
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id in vehicle_classes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicle_boxes.append((x1, y1, x2, y2))

    # Helper: Generate estimated parking spot coordinates across image rows
    def generate_parking_slots(width, height):
        slots = []
        # Define relative parking rows based on standard PKLot aerial perspectives
        row_configs = [
            {"y_start": 0.05, "height": 0.10, "count": 28},
            {"y_start": 0.22, "height": 0.12, "count": 28},
            {"y_start": 0.35, "height": 0.12, "count": 28},
            {"y_start": 0.55, "height": 0.14, "count": 26},
            {"y_start": 0.72, "height": 0.14, "count": 26},
        ]
        
        for config in row_configs:
            y1 = int(height * config["y_start"])
            y2 = int(y1 + (height * config["height"]))
            slot_w = width / config["count"]
            
            for i in range(config["count"]):
                x1 = int(i * slot_w + 2)
                x2 = int((i + 1) * slot_w - 2)
                slots.append((x1, y1, x2, y2))
                
        return slots

    slots = generate_parking_slots(img_w, img_h)

    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    occupied_count = 0
    empty_count = 0

    # Evaluate overlap between vehicles and parking slots
    for sx1, sy1, sx2, sy2 in slots:
        slot_area = (sx2 - sx1) * (sy2 - sy1)
        is_occupied = False

        for vx1, vy1, vx2, vy2 in vehicle_boxes:
            # Intersection coordinates
            ix1 = max(sx1, vx1)
            iy1 = max(sy1, vy1)
            ix2 = min(sx2, vx2)
            iy2 = min(sy2, vy2)

            if ix1 < ix2 and iy1 < iy2:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                if (intersection_area / slot_area) >= iou_threshold:
                    is_occupied = True
                    break

        if is_occupied:
            occupied_count += 1
            box_color = "#FF0000"  # Red for Occupied
        else:
            empty_count += 1
            box_color = "#00FF00"  # Green for Available

        draw.rectangle([sx1, sy1, sx2, sy2], outline=box_color, width=2)

    total_slots = len(slots)
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
        m1.metric("Total Slots Tracked", total_slots)
        m2.metric("Available Slots", empty_count)

        m3, m4 = st.columns(2)
        m3.metric("Occupied Slots", occupied_count)
        m4.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")

        st.markdown("---")
        st.subheader("Smart Guidance")
        st.markdown(f"**Congestion Level:** :{status_color}[{congestion_status}]")
        st.info(f"**Recommendation:** {recommendation}")
        
