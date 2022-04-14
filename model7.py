'''
Convolutional Neural Network Model to find faces in an image
Has 5 convolution layers followed by 3 linear layers
Modelled after AlexNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Convolutional Neural Network Model
# 5 convolution layers, followed by 3 fully connected layers
class FaceDetector7(nn.Module):

    def __init__(self):
        super().__init__()  # calling init of super class
        # first convolution - taking in 1 gray channels, outputting 16 channels
        # 11 x 11 kernel, stride 4
        # (200, 200, 1) -> (48, 48, 16)
        self.conv1 = nn.Conv2d(1, 16, 11, stride=4)
        # first max pool - downsampling by a factor of 2
        # (48, 48, 16) -> (24, 24, 16)
        self.pool1 = nn.MaxPool2d(2)
        # second convolution - taking in 16 channels, outputting 32 chanels
        # 5 x 5 kernel
        # (24, 24, 16) -> (20, 20, 32)
        self.conv2 = nn.Conv2d(16, 32, 5)
        # second max pool - downsampling by a factor of 2 x 2
        # (20, 20, 32) -> (10, 10, 32)
        self.pool2 = nn.MaxPool2d(2)
        # third convolution - taking in 32 channels, outputting 48 channels
        # 3 x 3 kernel
        # adding padding to keep same size outputs
        # (10, 10, 32) -> (10, 10, 48)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        # fourth convolution - taking in 48 channels, outputting 48 channels
        # 3 x 3 kernel
        # adding padding to keep same size outputs
        # (10, 10, 48) -> (10, 10, 48)
        self.conv4 = nn.Conv2d(48, 48, 3, padding=1)
        # fourth convolution - taking in 48 channels, outputting 32 channels
        # 3 x 3 kernel
        # adding padding to keep same size outputs
        # (10, 10, 48) -> (10, 10, 32)
        self.conv5 = nn.Conv2d(48, 32, 3, padding=1)
        # first fully connected layer
        # (3200) -> (3200)
        self.fc1 = nn.Linear(10 * 10 * 32, 3200)
        # second fully connected layer
        # (3200) -> (3200)
        self.fc2 = nn.Linear(3200, 3200)
        # third fully connected layer
        # (3200) -> (4)
        self.fc3 = nn.Linear(3200, 4)

    def forward(self, x):
        # relu units add non-linearity
        # 1st convolution, relu, pool
        x = self.pool1(F.relu(self.conv1(x)))
        # 2nd convolution, relu, pool
        x = self.pool2(F.relu(self.conv2(x)))
        # 3rd convolution, relu
        x = F.relu(self.conv3(x))
        # 4th convolution, relu
        x = F.relu(self.conv4(x))
        # 5th convolution, relu
        x = F.relu(self.conv5(x))
        # flatten all dimensions, except batch
        x = torch.flatten(x, 1)
        # first fully connected layer
        x = F.relu(self.fc1(x))
        # second fully connected layer
        x = F.relu(self.fc2(x))
        # third fully connected layer
        x = self.fc3(x)

        # regression output
        return x
