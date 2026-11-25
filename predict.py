from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train-4/weights/best.pt")

# Detect helmets
results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True
)

print("Helmet detection completed!")