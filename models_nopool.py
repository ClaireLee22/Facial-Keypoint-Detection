## define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


def init_weights(m):
    # initialize weights and bias
    if type(m) in [nn.Conv2d, nn.Linear]:
        #I.normal_(m.weight)
        I.xavier_normal_(m.weight)
        #I.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## 1st convolutional layer
        # output size = (224 - 5)/2 + 1 = 220
        # the output tensor will have dimensions: (32, 110, 110)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2)
        self.conv1.apply(init_weights)
        self.conv1_bn1 = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout(p=0.1)
    
    
        ## 2nd convolutional layer
        # output size = (110 - 3)/2 + 1 = 54
        # the output tensor will have dimensions: (64, 54, 54)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv2.apply(init_weights)
        self.conv2_bn2 = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout(p=0.2)
        
        ## 3rd convolutional layer
        # output size = (54 - 3)/2 + 1 = 26
        # the output tensor will have dimensions: (128, 26, 26)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv3.apply(init_weights)
        self.conv3_bn3 = nn.BatchNorm2d(128)
        self.conv3_drop = nn.Dropout(p=0.3)
        
        ## 4th convolutional layer
        # output size = (26 - 3)/2 + 1 = 12
        # the output tensor will have dimensions: (256, 12, 12)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv4.apply(init_weights)
        self.conv4_bn4 = nn.BatchNorm2d(256)
        self.conv4_drop = nn.Dropout(p=0.4)
        
        
        
        ## 1st fully connected layer
        # 512 input * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(256*12*12, 4096)
        self.fc1.apply(init_weights)
        self.fc1_bn1 = nn.BatchNorm1d(4096)
        self.fc1_drop = nn.Dropout(p=0.5)
        
        ## 2nd fully connected layer
        self.fc2 = nn.Linear(4096, 2048)
        self.fc2.apply(init_weights)
        self.fc2_bn2 = nn.BatchNorm1d(2048)
        self.fc2_drop = nn.Dropout(p=0.5)
        
        ## Output layer
        # finally, create 136 output channels
        self.fc3 = nn.Linear(2048, 136)
        self.fc3.apply(init_weights)
    

    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # four conv/relu + pool layers
        x = self.conv1_drop(F.relu(self.conv1_bn1(self.conv1(x))))
        #print('conv1', x.shape)
        x = self.conv2_drop(F.relu(self.conv2_bn2(self.conv2(x))))
        #print('conv2', x.shape)
        x = self.conv3_drop(F.relu(self.conv3_bn3(self.conv3(x))))
        #print('conv3', x.shape)
        x = self.conv4_drop(F.relu(self.conv4_bn4(self.conv4(x))))
        #print('conv4', x.shape)
        
       
        # reshape
        x = x.view(x.size(0), -1)
         
        # three linear layrers
        x = self.fc1_drop(F.relu(self.fc1_bn1(self.fc1(x))))
        x = self.fc2_drop(F.relu(self.fc2_bn2(self.fc2(x))))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

         
         
