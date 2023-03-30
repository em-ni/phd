import cv2
import torch
import numpy as np
from utils import get_loaders, plot_couple_examples, cells_to_bboxes, non_max_suppression
import config
from YOLOv3 import YOLOv3

def preprocess_frame(frame):
    frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

def draw_boxes(frame, detections, scaled_anchors, conf_threshold=0.6, iou_threshold=0.5):
    # just draw a rectangle for now
    x1 = 10
    y1 = 10
    x2 = 100
    y2 = 100
    cls_pred = 'test'
    conf = 0.0


    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{cls_pred} {conf*100:.2f}%", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)



    return frame

if __name__ == "__main__":
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    scaled_anchors = ( torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) ).to(config.DEVICE)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        with torch.no_grad():
            input_tensor = preprocess_frame(frame)
            detections = model(input_tensor)
            for i in range(len(detections)):
                print(detections[i].shape)
            draw_boxes(frame, detections, scaled_anchors)

        cv2.imshow("YOLOv3 Webcam", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
