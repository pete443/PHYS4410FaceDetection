'''
Loads images from DATA folder 
Use get_data_loader(batch_size) to get training and validation
dataloaders
'''

import os
import os.path
import sys
import numpy as np
import torch
import time
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
import glob
import random
import matplotlib.pyplot as plt

# DataSet class - subclassing PyTorch


class facesDataset(Dataset):

    # inputting pytorch tensors
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        # Denotes the total number of samples
        return self.X.size(dim=0)

    def __getitem__(self, index):

        return self.X[index], self.Y[index]

# returns training and validation data loaders for pytorch


def get_data_loader(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/train.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/val.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_data_loader_flip(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/trainFlip.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/valFlip.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_data_loader_gray_full(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/trainGray.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/valGray.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_data_loader_gray_flip(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/trainGrayFlip.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/valGrayFlip.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_data_loader_flip_norm(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/trainFlipNorm.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/valFlipNorm.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_data_loader200(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/train200.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/val200.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # turning to rgb - may not need
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_data_loader_gray(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/train.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/val.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # grayscaling image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    x_train = np.reshape(x_train, (1300, 100, 100, 1))
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # grayscaling image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    x_val = np.reshape(x_val, (400, 100, 100, 1))
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def get_data_loader_gray200flip(batch):
    # loading in image data
    DATA_DIR = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/"
    TRAIN_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/trainFlip200.txt"
    VAL_DATA_FILE = "/Users/petebuckman/Desktop/PHYS 4410/ML/DATA/FDDB/valFlip200.txt"

    # getting training data datasets
    with open(TRAIN_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    train_np_array_x = []
    train_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # grayscaling image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        train_np_array_x.append(img)
        train_np_array_y.append(label)

        i = i + 2

    x_train = np.array(train_np_array_x)
    x_train = np.reshape(x_train, (2720, 200, 200, 1))
    y_train = np.array(train_np_array_y)

    # getting validation data datasets
    with open(VAL_DATA_FILE) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    val_np_array_x = []
    val_np_array_y = []

    while i < len(lines):
        print(i)
        img_file_name = DATA_DIR + lines[i]
        rectangle = lines[i + 1].split()[0:4]
        x = float(rectangle[0])
        y = float(rectangle[1])
        w = float(rectangle[2])
        h = float(rectangle[3])
        label = np.array([x, y, w, h])

        # turning into numpy array
        img = cv2.imread(img_file_name)
        # grayscaling image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        val_np_array_x.append(img)
        val_np_array_y.append(label)

        i = i + 2

    x_val = np.array(val_np_array_x)
    x_val = np.reshape(x_val, (680, 200, 200, 1))
    y_val = np.array(val_np_array_y)

    # tests that we have the right data - can comment out later
    print(f"Shape of training data: {x_train.shape}")
    print(f"Shape of training labels: {y_train.shape}")
    print(f"Shape of validation data: {x_val.shape}")
    print(f"Shape of validation labels: {y_val.shape}")
    print(f"Train data data type: {type(x_train)}")
    print(f"Train labels data type: {type(y_train)}")
    print(f"Val data data type: {type(x_val)}")
    print(f"Val labels data type: {type(y_val)}")
    print(type(x_train[0][0][0][0]))
    print(type(y_train[0][0]))
    print(type(x_val[0][0][0][0]))
    print(type(y_val[0][0]))
    print(x_train[0][0][0][0])
    print(y_train[0][0])
    print(x_val[0][0][0][0])
    print(y_val[0][0])

    '''
    random_image = random.randint(0, len(x_train))
    plt.imshow(x_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()

    random_image = random.randint(0, len(x_val))
    plt.imshow(x_val[random_image])
    plt.title(f"Validation example #{random_image + 1300}")
    plt.axis('off')
    plt.show()
    '''

    # converting to pytorch datatypes
    # permuting for (C, H, W) format
    x_train_torch = torch.permute(
        torch.from_numpy(x_train), (0, 3, 1, 2)).float()
    y_train_torch = torch.from_numpy(y_train).float()
    x_val_torch = torch.permute(torch.from_numpy(x_val), (0, 3, 1, 2)).float()
    y_val_torch = torch.from_numpy(y_val).float()

    print(x_train_torch.size())

    # creating datasets
    train_dataset = facesDataset(x_train_torch, y_train_torch)
    val_dataset = facesDataset(x_val_torch, y_val_torch)

    # creating dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch, shuffle=True, num_workers=0)

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.show()


def show_batch(dl, nmax=64):

    for images in dl:
        show_images(images, nmax)
        break


def test():
    train_dataloader, val_dataloader = get_data_loader_gray(64)

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = torch.permute(train_features[0], (1, 2, 0))
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    print(f"Image: {img}")

    # Display image and label.
    val_features, val_labels = next(iter(val_dataloader))
    print(f"Feature batch shape: {val_features.size()}")
    print(f"Labels batch shape: {val_labels.size()}")
    img = torch.permute(val_features[0], (1, 2, 0))
    label = val_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    print(f"Image: {img}")


if __name__ == "__main__":
    test()
