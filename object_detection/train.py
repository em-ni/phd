"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

"""
Scaler Note: The scaler is used for mixed precision training in PyTorch. It is a GradScaler object that helps scale up the gradients computed during the backward pass so that 
they can be used to update the model's parameters. This is because some tensor cores in NVIDIA GPUs that perform the computations for training models work best with 
lower-precision data types like float16, but the optimizer's update step requires the gradients to be in float32.
To enable mixed precision training, the scaler is used in conjunction with PyTorch's automatic mixed precision (AMP) feature. 
The AMP feature dynamically casts the model's parameters and activations to a lower-precision data type like float16 while keeping the optimizer's parameters and gradients in float32,
resulting in faster training without sacrificing accuracy.

Anchors Note: In the context of object detection using the YOLOv3 model, anchors are a set of predefined bounding boxes with specific width and height ratios that are used to 
predict the location and size of objects in an image. The scaled anchors are a tensor that contains the width and height of each anchor box, scaled by the grid cell size in the 
YOLOv3 output feature maps.
The YOLOv3 model predicts object bounding boxes at three different scales (on three different feature maps), and each scale uses a different set of anchors. 
The anchor boxes are specified in the model configuration file config.py as a list of tuples, with each tuple representing the width and height of an anchor box.
To create the scaled anchors, the anchor widths and heights are first multiplied by the corresponding stride values (config.S), which determine the size of the grid cells in 
each feature map. The resulting tensor has shape (3, num_anchors_per_scale, 2), where the first dimension represents the scale and the second dimension represents the anchor box index.
The third dimension contains the scaled width and height values of each anchor box.
The scaled anchors are then moved to the same device (CPU or GPU) as the model using the .to() method and are passed to the train_fn() function, where they are used to compute the YOLOv3 loss function for each scale.
"""

import config
import torch
import torch.optim as optim

from YOLOv3 import YOLOv3
from YOLOv3Tiny import YOLOv3Tiny
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)



def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    #model = YOLOv3Tiny(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = ( torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        print(f"\nCurrent epoch: {epoch}")

        # Train
        print("Training...")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        # Check class accuracy on train set        
        print("Checking class accuracy on Train loader:")
        check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        # Every 3 epochs check mAP on test set
        if epoch > 0 and epoch % 3 == 0:
            print("Checking class accuracy on Test loader:")
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD)
            mapval = mean_average_precision(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH, box_format="midpoint", num_classes=config.NUM_CLASSES)
            print(f"MAP: {mapval.item()}")

            # Set model back to train mode (was in eval mode in check_class_accuracy and get_evaluation_bboxes)
            model.train()


if __name__ == "__main__":
    main()
