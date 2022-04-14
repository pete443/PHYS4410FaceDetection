'''
Convolutional Neural Network Model to find faces in an image
Has 4 convolution layers followed by 2 linear layers
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Convolutional Neural Network Model
# 4 convolution layers, followed by 2 fully connected layers
class FaceDetector2(nn.Module):

    def __init__(self):
        super().__init__()  # calling init of super class
        # first convolution - taking in 3 image channels, outputting 32 channels
        # 3 x 3 kernel
        # (100, 100, 3) -> (98, 98, 32)
        self.conv1 = nn.Conv2d(3, 32, 3)
        # first max pool - downsampling by a factor of 2
        # (98, 98, 32) -> (49, 49, 32)
        self.pool1 = nn.MaxPool2d(2)
        # second convolution - taking in 32 image channels, outputting 32 chanels
        # 3 x 3 kernel
        # (49, 49, 32) -> (47, 47, 32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        # second max pool - downsampling by a factor of 2 x 2
        # (47, 47, 32) -> (23, 23, 32)
        self.pool2 = nn.MaxPool2d(2)
        # third convolution - taking in 32 image channels, outputting 32 channels
        # 3 x 3 kernel
        # (23, 23, 32) -> (21, 21, 32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        # third max pool - downsampling by a factor of 2 x 2
        # (21, 21, 32) -> (10, 10, 32)
        self.pool3 = nn.MaxPool2d(2)
        # fourth convolution - taking in 32 image channels, outputting 32 channels
        # 3 x 3 kernel
        # (10, 10, 32) -> (8, 8, 32)
        self.conv4 = nn.Conv2d(32, 32, 3)
        # fourth max pool - downsampling by a factor of 2 x 2
        # (8, 8, 32) -> (4, 4, 32)
        self.pool4 = nn.MaxPool2d(2)
        # first fully connected layer
        # (512) -> (256)
        self.fc1 = nn.Linear(4 * 4 * 32, 64)
        # second fully connected layer
        # (64) -> (4)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        # relu units add non-linearity
        # 1st convolution, relu, pool
        x = self.pool1(F.relu(self.conv1(x)))
        # 2nd convolution, relu, pool
        x = self.pool2(F.relu(self.conv2(x)))
        # 3rd convolution, relu, pool
        x = self.pool3(F.relu(self.conv3(x)))
        # 4th convolution, relu, pool
        x = self.pool4(F.relu(self.conv4(x)))
        # flatten all dimensions, except batch
        x = torch.flatten(x, 1)
        # first fully connected layer
        x = F.relu(self.fc1(x))
        # second fully connected layer - don't relu for regression outputs
        x = self.fc2(x)

        # regression output
        return x
