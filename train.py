from ultralytics import YOLO

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="Helmet - No helmet detection.v1i.yolov8/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    device="cpu"
)