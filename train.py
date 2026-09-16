import os
from ultralytics import YOLO

def run_training():
    # Initialize base YOLO model
    model = YOLO("yolov8n.pt")

    # Train model using dataset configuration
    results = model.train(
        data="dataset/data.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        name="parkvision_run",
        project="runs/detect"
    )

    # Perform validation evaluation
    metrics = model.val()
    print(f"Validation mAP50 Metric: {metrics.box.map50:.4f}")
    print("Training finished successfully. Saved weights to runs/detect/parkvision_run/weights/best.pt")

if __name__ == "__main__":
    run_training()
