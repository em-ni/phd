"""
ConvLayer dimension note: 
    After a convolutional layer in a neural network, the dimensions of the output layer can be calculated using the following formula:

    output_height = (input_height - filter_height + 2 * padding) / stride + 1
    output_width = (input_width - filter_width + 2 * padding) / stride + 1

    where:
    -input_height and input_width are the dimensions of the input feature map
    -filter_height and filter_width are the dimensions of the convolutional filter/kernel
    -padding is the number of pixels added to the border of the input feature map before applying the filter. There are two types of padding: 'valid' padding (no padding) 
    and 'same' padding (padding such that the output feature map has the same spatial dimensions as the input feature map)
    -stride is the number of pixels by which the filter/kernel is moved across the input feature map

    The output of the convolutional layer will have the same number of channels as the number of filters applied in that layer.

    For example, if the input feature map has dimensions (32, 32, 3) (height, width, number of channels), and the convolutional layer has 64 filters of size 3x3, stride of 1, 
    and same padding, then the output feature map will have dimensions (32, 32, 64).
    It is important to keep track of the spatial dimensions of the feature maps throughout the neural network, especially when designing the architecture of the network and 
    defining the input and output shapes of each layer.
"""

# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
#1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    for epoch in range(10): # train for 10 epochs
        for batch in dataset: 
            X,y = batch 
            X, y = X.to('cuda'), y.to('cuda') 
            yhat = clf(X) 
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            opt.zero_grad()
            loss.backward() 
            opt.step() 

        print(f"Epoch:{epoch} loss is {loss.item()}")
    
    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open('img_3.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))