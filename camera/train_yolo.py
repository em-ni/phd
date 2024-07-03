# Does not work even with the smallest model yolov10n.pt

from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("yolov10n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=100, imgsz=640, batch=16)