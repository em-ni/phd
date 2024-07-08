from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10x.pt")

# Perform object detection on an image
im2 = cv2.imread("C:\\Users\\z5440219\\OneDrive - UNSW\Desktop\\github\\surgical-copilot\\camera\\object_detection_YOLOv3\\test_images\\people.jpg")
results = model.predict(source=im2, save=True, save_txt=True)