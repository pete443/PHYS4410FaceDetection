'''
Class to run the model on test images

'''

import torch
from model1 import FaceDetector1
from model2 import FaceDetector2
from model3 import FaceDetector3
from model4 import FaceDetector4
from model5 import FaceDetector5
from model7 import FaceDetector7
from datagen import get_data_loader
from datagen import get_data_loader_flip
from datagen import get_data_loader200
from datagen import get_data_loader_gray
from datagen import get_data_loader_gray200flip
from main import calc_loss
import cv2
import numpy as np
import os

if __name__ == "__main__":

    # loading PyTorch model
    model = FaceDetector3()
    model_name = "models/model3_lr=0.00001_b=0.99_sched=0.5_epochs=30_flip"
    model.load_state_dict(torch.load(model_name))

    choice = 4
    pixels = 100

    if choice == 0:
        train_dataloader, val_dataloader = get_data_loader(32)
    elif choice == 1:
        train_dataloader, val_dataloader = get_data_loader200(32)
    elif choice == 2:
        train_dataloader, val_dataloader = get_data_loader_gray(32)
    elif choice == 3:
        train_dataloader, val_dataloader = get_data_loader_gray200flip(32)
    elif choice == 4:
        train_dataloader, val_dataloader = get_data_loader_flip(32)

    # evaluating on test images (using a batch)

    for img_file in os.listdir("test_images"):
        print("test_images/" + img_file)
        if img_file.endswith(".jpg"):

            img = cv2.imread("test_images/" + img_file)
            height, width, channels = img.shape
            dsize = (pixels, pixels)
            img2 = cv2.resize(img, dsize)

            # turning into pyTorch array
            if choice == 2 or choice == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                temp = np.reshape(np.array(img2), (200, 200, 1))
                input = torch.from_numpy(temp).float()
                input = torch.permute(input, (2, 0, 1))
                input = torch.reshape(input, (1, 1, pixels, pixels))
            else:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                input = torch.from_numpy(np.array(img2)).float()
                input = torch.permute(input, (2, 0, 1))
                input = torch.reshape(input, (1, 3, pixels, pixels))

            # running through model
            output = model(input)
            x = round(output[0, 0].item() * width / pixels)
            y = round(output[0, 1].item() * height / pixels)
            w = round(output[0, 2].item() * width / pixels)
            h = round(output[0, 3].item() * height / pixels)

            # drawing the box on the image
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 255, 0), round(height / 300))

            cv2.imwrite("results/" + img_file, img)

    # printing validation loss
    val_loss = calc_loss(model, val_dataloader)
    print("Validation loss: " + str(val_loss))
