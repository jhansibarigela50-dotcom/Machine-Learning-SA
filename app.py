"""
ParkVision AI — Smart Parking Occupancy Dashboard
Loads a trained YOLO model and detects occupied/empty parking spaces
in an uploaded image, then shows occupancy stats and congestion insights.
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import time

# ---------- Page config ----------
st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🚗",
    layout="wide",
)

# ---------- Constants ----------
MODEL_PATH = "model/parkvision_yolo_best_v2.pt"
CONF_THRESHOLD_DEFAULT = 0.4

# ---------- Load model (cached so it only loads once) ----------
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

# Figure out which class names mean "occupied" vs "empty" automatically,
# so this works regardless of exact label spelling (e.g. "space-occupied", "occupied", "car").
def classify_names(names_dict):
    occupied_ids, empty_ids = [], []
    for idx, name in names_dict.items():
        lname = name.lower()
        if "occup" in lname or "car" in lname or "busy" in lname:
            occupied_ids.append(idx)
        elif "empty" in lname or "free" in lname or "vacant" in lname:
            empty_ids.append(idx)
    return occupied_ids, empty_ids

occupied_ids, empty_ids = classify_names(model.names)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, CONF_THRESHOLD_DEFAULT, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model classes detected:**")
st.sidebar.json(model.names)

# ---------- Header ----------
st.title("🚗 ParkVision AI")
st.caption("Smart parking occupancy detection — upload a parking lot image to see live slot status.")

# ---------- Input ----------
uploaded_file = st.file_uploader("Upload a parking lot image", type=["jpg", "jpeg", "png"])

col_img, col_stats = st.columns([2, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("Running detection..."):
        start = time.time()
        results = model.predict(img_array, conf=conf_threshold, verbose=False)
        elapsed = time.time() - start

    result = results[0]
    annotated = result.plot()  # BGR numpy array with boxes drawn
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Count detections per class
    class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
    n_occupied = sum(1 for c in class_ids if c in occupied_ids)
    n_empty = sum(1 for c in class_ids if c in empty_ids)
    n_total = n_occupied + n_empty

    with col_img:
        st.image(annotated_rgb, caption=f"Detections ({elapsed:.2f}s)", use_column_width=True)

    with col_stats:
        st.subheader("📊 Occupancy Summary")
        st.metric("Total spaces detected", n_total)
        st.metric("Occupied", n_occupied)
        st.metric("Available", n_empty)

        if n_total > 0:
            occupancy_rate = n_occupied / n_total * 100
            st.progress(min(int(occupancy_rate), 100))
            st.write(f"**Occupancy rate:** {occupancy_rate:.1f}%")

            # Congestion insight
            if occupancy_rate >= 90:
                st.error("🔴 High congestion — lot nearly full")
            elif occupancy_rate >= 60:
                st.warning("🟠 Moderate congestion")
            else:
                st.success("🟢 Low congestion — plenty of space")

            chart_df = pd.DataFrame({
                "Status": ["Occupied", "Available"],
                "Count": [n_occupied, n_empty]
            })
            st.bar_chart(chart_df.set_index("Status"))
        else:
            st.info("No parking spaces detected. Try lowering the confidence threshold.")

else:
    st.info("👆 Upload a parking lot image to get started.")

st.markdown("---")
st.caption("ParkVision AI · Trained on the PKLot dataset · Built with YOLO + Streamlit")
