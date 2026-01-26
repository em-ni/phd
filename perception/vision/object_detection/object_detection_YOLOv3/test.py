"""
Test the model on a couple of examples from the test set
"""

import torch
import numpy as np
from utils import get_loaders, plot_couple_examples
import config
from YOLOv3 import YOLOv3


if __name__ == "__main__":
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    train_loader, test_loader, train_eval_loader = get_loaders(train_csv_path=config.DATASET + "/8examples.csv", test_csv_path=config.DATASET + "/8examples.csv")
    scaled_anchors = ( torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) ).to(config.DEVICE)
    plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)

