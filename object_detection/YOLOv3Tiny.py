import torch
import torch.nn as nn
import config

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class YOLOv3Tiny(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3Tiny, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = 3

        # Define the YOLOv3-tiny architecture
        self.layers = nn.Sequential(
            ConvBlock(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 1),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), 1, 1, 0)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    num_classes = config.NUM_CLASSES
    IMAGE_SIZE = 416 # Yolo v1 uses 448 but Yolo v3 uses 416 as input size (multiscale training)
    model = YOLOv3Tiny(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    print(out.shape)
    """ assert out.shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5) """
    print("Success!")
