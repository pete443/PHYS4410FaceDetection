'''
Convolutional Neural Network Model to find faces in an image
Has 4 convolution layers followed by 4 linear layers.
Uses gray-scale images.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Convolutional Neural Network Model
# 4 convolution layers, followed by 4 fully connected layers
class FaceDetector4(nn.Module):

    def __init__(self):
        super().__init__()  # calling init of super class
        # first convolution - taking in 3 image channels, outputting 32 channels
        # 5 x 5 kernel
        # (200, 200, 3) -> (196, 196, 32)
        self.conv1 = nn.Conv2d(3, 32, 5)
        # first max pool - downsampling by a factor of 2
        # (196, 196, 32) -> (98, 98, 32)
        self.pool1 = nn.MaxPool2d(2)
        # second convolution - taking in 32 image channels, outputting 32 chanels
        # 5 x 5 kernel
        # (98, 98, 32) -> (94, 94, 32)
        self.conv2 = nn.Conv2d(32, 32, 5)
        # second max pool - downsampling by a factor of 2 x 2
        # (94, 94, 32) -> (47, 47, 32)
        self.pool2 = nn.MaxPool2d(2)
        # third convolution - taking in 32 image channels, outputting 32 channels
        # 5 x 5 kernel
        # (47, 47, 32) -> (43, 43, 32)
        self.conv3 = nn.Conv2d(32, 32, 5)
        # third max pool - downsampling by a factor of 2 x 2
        # (43, 43, 32) -> (21, 21, 32)
        self.pool3 = nn.MaxPool2d(2)
        # fourth convolution - taking in 32 image channels, outputting 32 channels
        # 5 x 5 kernel
        # (21, 21, 32) -> (17, 17, 32)
        self.conv4 = nn.Conv2d(32, 32, 5)
        # fourth max pool - downsampling by a factor of 2 x 2
        # (17, 17, 32) -> (8, 8, 32)
        self.pool4 = nn.MaxPool2d(2)
        # first fully connected layer
        # (2048) -> (512)
        self.fc1 = nn.Linear(8 * 8 * 32, 512)
        # second fully connected layer
        # (512) -> (128)
        self.fc2 = nn.Linear(512, 128)
        # third fully connected layer
        # (128) -> (32)
        self.fc3 = nn.Linear(128, 32)
        # fourth fully connected layer
        # (32) -> (4)
        self.fc4 = nn.Linear(32, 4)

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
        # second fully connected layer
        x = F.relu(self.fc2(x))
        # third fully connected layer
        x = F.relu(self.fc3(x))
        # fourth fully connected layer
        x = self.fc4(x)

        # regression output
        return x
