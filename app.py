"""
ParkVision AI - Intelligent Urban Parking Analytics & Space Optimisation
Streamlit web app that runs a trained YOLOv8 model on an uploaded parking
lot image, marks each slot as Occupied / Empty, and shows occupancy stats
and a recommendation.

Run locally:
    streamlit run app.py
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --------------------------------------------------------------------------
# CONFIG — edit these to match how you trained your model
# --------------------------------------------------------------------------
MODEL_PATH = "best.pt"

# IMPORTANT: this must match the class order your model was trained with.
# Check it yourself in Colab with:  print(model.names)
# and edit this dict so the keys/order line up exactly.
CLASS_NAMES = {
    0: "empty",
    1: "occupied",
}

CONFIDENCE_THRESHOLD = 0.4

# Colors are in BGR (OpenCV format)
COLOR_EMPTY = (0, 200, 0)       # green
COLOR_OCCUPIED = (0, 0, 230)    # red

# --------------------------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🅿️",
    layout="wide",
)

st.title("🅿️ ParkVision AI")
st.caption("Intelligent Urban Parking Analytics & Space Optimisation")


# --------------------------------------------------------------------------
# MODEL LOADING (cached so it only loads once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def get_label(class_id: int) -> str:
    """Map a class id to a normalized label, defaulting sensibly if unknown."""
    name = CLASS_NAMES.get(class_id, "unknown").lower()
    if "occ" in name or name in ("car", "vehicle", "busy"):
        return "occupied"
    return "empty"


def run_inference(model, image: Image.Image):
    """Run YOLO on a PIL image, return (annotated_bgr_image, occupied_count, empty_count)."""
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(img_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result = results[0]

    occupied_count = 0
    empty_count = 0

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = get_label(cls_id)
        if label == "occupied":
            occupied_count += 1
            color = COLOR_OCCUPIED
        else:
            empty_count += 1
            color = COLOR_EMPTY

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img_bgr, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return img_bgr, occupied_count, empty_count


def congestion_level(occupancy_pct: float) -> str:
    if occupancy_pct < 40:
        return "Low"
    elif occupancy_pct <= 75:
        return "Moderate"
    else:
        return "High"


def recommendation(occupancy_pct: float, available: int) -> str:
    if available == 0:
        return "🚫 Parking full — try another area."
    if occupancy_pct > 75:
        return "⚠️ Parking is nearly full — consider another location if you need guaranteed space."
    return "✅ Slots available — proceed to this parking lot."


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "Upload a parking lot image and ParkVision AI will detect each "
        "slot, mark it as **occupied** or **empty**, and summarise "
        "availability."
    )
    st.markdown("---")
    st.write(f"Confidence threshold: **{CONFIDENCE_THRESHOLD}**")


# --------------------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a parking lot image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(
            f"Could not load the model from `{MODEL_PATH}`. "
            f"Make sure your trained weights file is committed to the repo "
