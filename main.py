'''
Training the CNN
'''

from model1 import FaceDetector1
from model2 import FaceDetector2
from model3 import FaceDetector3
from model4 import FaceDetector4
from model5 import FaceDetector5
from model6 import FaceDetector6
from model7 import FaceDetector7
from datagen import get_data_loader
from datagen import get_data_loader_flip
from datagen import get_data_loader_gray_full
from datagen import get_data_loader_gray_flip
from datagen import get_data_loader200
from datagen import get_data_loader_gray
from datagen import get_data_loader_gray200flip
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import numpy as np


# to initialize weights with xavier uniform
def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_uniform(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

# to calculate error


# to calculate average loss over training and validation sets
def calc_loss(model, loader):
    size = loader.dataset.__len__()
    criterion = nn.MSELoss()
    iter = math.ceil(size / loader.batch_size)

    # taking a mean loss over all batches
    running_loss = 0.0
    for i, (input, target) in enumerate(loader):
        output = model(input)
        running_loss += criterion(output, target).item()

    # loss averaged over batch sizes
    return running_loss / iter


# training loop
def train(model_path, train_dataloader, val_dataloader, step, mom, epochs):

    # training loss list
    train_loss_list = []
    # validation loss list
    val_loss_list = []

    # initialize model
    model = FaceDetector1()
    # initializing weights
    model.apply(init_weights_xavier)

    # defining loss as mean squared error
    criterion = nn.MSELoss()
    # SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=step, momentum=mom)
    # scheduler to adjust learning rate
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 10, 15, 20, 25, 30, 35, 40], gamma=0.5)

    print("Training Model...")

    # initial losses
    train_loss_list.append(calc_loss(model, train_dataloader))
    val_loss_list.append(calc_loss(model, val_dataloader))

    total_time = 0  # calculating training time
    lowest_val_loss = 100000  # saving lowest validation loss model
    for epoch in range(epochs):
        start = time.time()

        print("Epoch: " + str(epoch) + " running...")

        for i, (input, target) in enumerate(train_dataloader):
            # clearing gradient
            optimizer.zero_grad()
            # forward pass
            output = model(input)
            # calculating loss
            loss = criterion(output, target)

            print("Batch " + str(i) + " loss: " + str(loss.item()))

            # calculating derivative of loss wrt parameters
            loss.backward()
            # taking the next step
            optimizer.step()

            # stepping the learning rate
            scheduler.step()

        end = time.time()

        # calculating average loss
        train_loss = calc_loss(model, train_dataloader)
        val_loss = calc_loss(model, val_dataloader)

        print("Epoch: " + str(epoch) + " time: " + str(end - start))
        print("Training loss: " + str(train_loss))
        print("Validation loss: " + str(val_loss))

        # adding loss to lists
        train_loss_list.append(calc_loss(model, train_dataloader))
        val_loss_list.append(calc_loss(model, val_dataloader))

        # saving best model
        if epoch >= 10 and val_loss < lowest_val_loss:
            print("Saving model")
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        # not including time to calculate loss and error
        total_time += (end - start)

    print("Done training!")

    return model, total_time, train_loss_list, val_loss_list


# main function call
if __name__ == "__main__":

    switch = 0

    if switch == 0:
        # getting data loaders
        train_dataloader, val_dataloader = get_data_loader(32)

        # name the model
        model_name = "model1_lr=0.00001_b=0.99_sched=0.5_epochs=20"

        # finding relevant directories
        model_dir = "models"
        train_loss_dir = "train_loss"
        val_loss_dir = "val_loss"
        model_path = model_dir + "/" + model_name
        train_loss_path = train_loss_dir + "/" + model_name + ".txt"
        val_loss_path = val_loss_dir + "/" + model_name + ".txt"
        train_time_path = "times/" + model_name + ".txt"

        # training network
        model, total_time, train_loss_list, val_loss_list = train(
            model_path, train_dataloader, val_dataloader, 0.00001, 0.99, 20)
        print("Total training time: " + str(total_time))

        # saving losses
        with open(train_loss_path, 'w') as f:
            for i in range(len(train_loss_list)):
                f.write(str(i) + " " + str(train_loss_list[i]) + "\n")

            f.close()

        with open(val_loss_path, 'w') as f:
            for i in range(len(val_loss_list)):
                f.write(str(i) + " " + str(val_loss_list[i]) + "\n")

            f.close()

        # saving training times
        with open(train_time_path, 'w') as f:
            f.write(str(total_time) + "\n")

            f.close()
