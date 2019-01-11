## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # we start with the 32 feature maps and a 5x Kernel
        # we are changing to 4x4 Kenel 
        #self.conv1 = nn.Conv2d(1, 32, 5)
        #self.conv1 = nn.Conv2d(1, 32, 4)
        #self.conv1 = nn.Conv2d(1, 32, 7)
        self.conv1 = nn.Conv2d(1, 32, 5)

        ## secod layer we have 32 depth and we want 64 feature maps but reduce Kernel to 3x3
        self.conv2 = nn.Conv2d(32, 64, 3)
        # third conv layer: 64 inputs, 128 feature maps, 3x3 Kernel
        self.conv3 = nn.Conv2d(64, 128, 3)
        # we tried 2x2 for this layer but not much enhancements , going back to 3x3
        #self.conv3 = nn.Conv2d(64, 128, 2)
        # final fourth layer with 256 feature maps and 2x2 Kernel
        #self.conv4 = nn.Conv2d(128,256,2)
        # going back to 3x3
        self.conv4 = nn.Conv2d(128,256,3)
        ## adding a 5th Conv layer with 522 feature maps
        ##self.conv5 = nn.Conv2d(256,512,2)
        
        # apply max pooling with stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # full connected layer with 136 output 2 for each of the key pairs
        
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000)
        #self.fc1 = nn.Linear(in_features = 12800, out_features = 1000)
        
        self.fc2 = nn.Linear(in_features = 1000 , out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000 , out_features = 1000)
        self.fc4 = nn.Linear(in_features = 1000 , out_features = 1000)
        self.fc5 = nn.Linear(in_features = 1000 , out_features = 136)
        
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        ## x = self.drop1(self.pool(F.relu(self.conv1(x))))
        #x = self.drop1(self.pool(F.elu(self.conv1(x))))
        #x = self.drop1(self.pool(F.selu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        ##x = self.drop2(self.pool(F.relu(self.conv2(x))))
        #x = self.drop2(self.pool(F.elu(self.conv2(x))))
        #x = self.drop2(self.pool(F.selu(self.conv2(x))))
        x = self.pool(F.relu(self.conv3(x)))
        ##x = self.drop3(self.pool(F.relu(self.conv3(x))))
        #x = self.drop3(self.pool(F.elu(self.conv3(x))))
        x = self.pool(F.relu(self.conv4(x)))
        ##x = self.drop4(self.pool(F.relu(self.conv4(x))))
        #x = self.drop4(self.pool(F.elu(self.conv4(x))))
        
        #x = self.drop5(self.pool(F.relu(self.conv5(x))))
        #x = self.drop4(self.pool(F.elu(self.conv5(x))))
        # Flatten the layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        ## x = self.drop5(F.relu(self.fc1(x)))
        ##x = self.drop5(F.relu(self.fc2(x)))
        x = F.relu(self.fc2(x))
        ## added
        ##x = self.drop5(F.relu(self.fc3(x)))
        ##x = self.drop6(F.relu(self.fc4(x)))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        ##
        #x = self.fc3(x)
        x = self.fc5(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
