from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10x.pt")

# Perform object detection on an image
im2 = cv2.imread("data/fun/standing.jpg")
results = model.predict(source=im2, save=True, save_txt=True)