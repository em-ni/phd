"""
Performs real-time object detection on a webcam feed using a trained YOLOv3 model.
"""

import cv2
import torch
import numpy as np
from utils import cells_to_bboxes, non_max_suppression
import config
from YOLOv3 import YOLOv3

def preprocess_frame(frame):
    """
    Preprocesses the frame before passing it to the model.
    """
    # Resize to 416x416
    frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE))

    # Converts the pixel values of the resized image to 32-bit floating-point.
    # Then normalizes the pixel values by dividing them by 255.0, which scales the values from the original 0 to 255 range to a new 0.0 to 1.0 range.
    frame = frame.astype(np.float32) / 255.0

    # Converts the image from HWC (height, width, channels) to CHW (channels, height, width) format.
    frame = np.transpose(frame, (2, 0, 1))

    # Adds a new axis to the image, which is the batch dimension (unsqeeze) and move it to the GPU.
    return torch.tensor(frame, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)


def get_boxes(y, scaled_anchors, conf_threshold=0.6, iou_threshold=0.5):
    """
    Converts the model's predictions into a list of bounding boxes.
    Inputs:
        y: the model's predictions
        scaled_anchors: the scaled anchor boxes
        conf_threshold: the confidence threshold
        iou_threshold: the IoU threshold
    Returns:
        A list of bounding boxes, each in the format [class_pred, conf, x, y, width, height]
    """

    boxes = []

    # For each anchor box  
    for i in range(y[0].shape[1]): 
        anchor = scaled_anchors[i]
        # Converts the model's predictions for the current anchor box and the current prediction into a list of bounding boxes.
        boxes += cells_to_bboxes( y[i], is_preds=True, S=y[i].shape[2], anchors=anchor)[0]
    
    # Applies non-max suppression to the bounding boxes.
    boxes = non_max_suppression(boxes, iou_threshold=iou_threshold, threshold=conf_threshold, box_format="midpoint")
    
    return boxes

def draw_bb(image, boxes, class_labels, colors):
    """
    Draws bounding boxes and class labels on the image.
    Inputs:
        image: the image to draw the bounding boxes and labels on
        boxes: a list of bounding boxes in the format [class_pred, conf, x, y, width, height]
        class_labels: a list of class labels
        colors: a list of colors for each class
    Returns:
        The image with bounding boxes and labels drawn on it.
    """
    
    # Get image dimensions, discard the channels dimension
    height, width, _ = image.shape

    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height in this order"
        # Get data from box and convert to pixel values (box = [class_pred, conf, x, y, width, height])
        class_pred = int(box[0])
        box = box[2:]
        upper_left_x = (box[0] - box[2] / 2) * width
        upper_left_y = (box[1] - box[3] / 2) * height
        lower_right_x = (box[0] + box[2] / 2) * width
        lower_right_y = (box[1] + box[3] / 2) * height

        # Draw bounding box and class label with confidence
        cv2.rectangle(image, (int(upper_left_x), int(upper_left_y)), (int(lower_right_x), int(lower_right_y)), colors[class_pred], 2)
        text_x = int(upper_left_x + (lower_right_x - upper_left_x) / 2)
        text_y = int(upper_left_y + (lower_right_y - upper_left_y) / 2)
        cv2.putText(image, class_labels[class_pred] + " " + str(box[1]) + "%", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_pred], 2)

    return image

if __name__ == "__main__":
    # Load saved model
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Get scaled anchors as a tensor of shape (3, 3, 2) where 3 is the number of anchor boxes and 2 is the width and height of each anchor box.
    scaled_anchors = ( torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) ).to(config.DEVICE)

    # get class labels and generate random colors for each class
    # TODO - add personal class labels
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype=int).tolist()

    # Test with webcam or image
    web = True
    if web == True:
        # Test with webcam
        cap = cv2.VideoCapture(0)
        while True:
            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            with torch.no_grad(): # Don't need to track gradients for inference, so save memory
                # Preprocess frame to tensor
                input_tensor = preprocess_frame(frame)

                # Get model predictions
                y = model(input_tensor)

                # Get bounding boxes (apply non-max suppression)
                boxes = get_boxes(y, scaled_anchors, conf_threshold=0.7, iou_threshold=0.7)

                # Draw bounding boxes on frame
                frame_bb = draw_bb(frame, boxes, class_labels, colors)

            cv2.imshow("YOLOv3 Webcam", frame_bb)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        # Test with a single image
        img = cv2.imread("people.jpg")
        cv2.imshow("before prediction", img)
        input_tensor = preprocess_frame(img)
        y = model(input_tensor)
        boxes = get_boxes(y, scaled_anchors)
        image_bb = draw_bb(img, boxes, class_labels, colors)
        cv2.imshow("opencv bb image", image_bb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
