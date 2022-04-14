'''
Convolutional Neural Network Model to find faces in an image
Has 3 convolution layers followed by 3 linear layers.
Uses gray scale images
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Convolutional Neural Network Model
# 3 convolution layers, followed by 3 fully connected layers
class FaceDetector6(nn.Module):

    def __init__(self):
        super().__init__()  # calling init of super class
        # first convolution - taking in 3 image channels, outputting 16 channels
        # 5 x 5 kernel
        # (100, 100, 3) -> (96, 96, 16)
        self.conv1 = nn.Conv2d(1, 16, 5)
        # first max pool - downsampling by a factor of 2
        # (96, 96, 16) -> (48, 48, 16)
        self.pool1 = nn.MaxPool2d(2)
        # second convolution - taking in 16 image channels, outputting 32 chanels
        # 3 x 3 kernel
        # (48, 48, 16) -> (46, 46, 32)
        self.conv2 = nn.Conv2d(16, 32, 3)
        # second max pool - downsampling by a factor of 2 x 2
        # (46, 46, 32) -> (23, 23, 32)
        self.pool2 = nn.MaxPool2d(2)
        # third convolution - taking in 32 image channels, outputting 32 channels
        # 3 x 3 kernel
        # (23, 23, 32) -> (21, 21, 32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        # third max pool - downsampling by a factor of 2 x 2
        # (21, 21, 32) -> (10, 10, 32)
        self.pool3 = nn.MaxPool2d(2)
        # first fully connected layer
        # (3200) -> (256)
        self.fc1 = nn.Linear(10 * 10 * 32, 256)
        # second fully connected layer
        # (256) -> (32)
        self.fc2 = nn.Linear(256, 32)
        # third fully connected layer
        # (32) -> (4)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        # relu units add non-linearity
        # 1st convolution, relu, pool
        x = self.pool1(F.relu(self.conv1(x)))
        # 2nd convolution, relu, pool
        x = self.pool2(F.relu(self.conv2(x)))
        # 3rd convolution, relu, pool
        x = self.pool3(F.relu(self.conv3(x)))
        # flatten all dimensions, except batch
        x = torch.flatten(x, 1)
        # first fully connected layer
        x = F.relu(self.fc1(x))
        # second fully connected layer
        x = F.relu(self.fc2(x))
        # 3rd fully connected layer - don't relu for regression outputs
        x = self.fc3(x)

        # regression output
        return x
