# Inferencing: Load saved model and inference a sample photo
# 11/17/22

import os
import requests
import zipfile
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torchinfo import summary
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from joblib.externals.loky.backend.context import get_context

# model location
MODEL_PATH = Path("models")
MODEL_NAME = "electronic_components-9864-6963.pth"   # train_acc, test_acc
MODEL_PATH_NAME = MODEL_PATH / MODEL_NAME

BATCH_SIZE = 32
NODES = 64

class_names = ['IC_chip', 'TO-3_transistor', 'TO-92_transistor', 'ceramic_capacitor', 'electrolytic_capacitor',  'metal_film_resistor', 'relay']

# load a sample unknown image
# filepathname = "./unknown/resistor1.jpg"
filepathname = "./unknown/resistors2.jpg"
filepathname = "./unknown/resistor3.jpg"
filepathname = "./unknown/transistor1.jpg"
filepathname = "./unknown/transistor2.jpg"
filepathname = "./unknown/transistor4.jpg"
filepathname = "./unknown/power-trans1.jpg"
filepathname = "./unknown/ceramic1.jpg"
filepathname = "./unknown/ceramic2.jpg"
filepathname = "./unknown/ceramic3.jpg" # metal film resistor
# filepathname = "./unknown/electrolytics1.jpg"
# filepathname = "./unknown/electro2.jpg"
# filepathname = "./unknown/electro3.jpg" # metal film resistor
# filepathname = "./unknown/electro4.jpg"
# filepathname = "./unknown/chip1.jpg"
filepathname = "./unknown/chip3.jpg"
filepathname = "./unknown/relay1.jpg"
filepathname = "./unknown/relay2.jpg"
filepathname = "./unknown/relay3.jpg"


# setup cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"NOTE: All calculations will be done in {device}")

# load state_dict and assign to model


data_transform = transforms.Compose([  # Compose is used to serialize our tranforms functions
    # transform operations are a list
    # 1. resize image to be smaller
    transforms.Resize(size=(64, 64)),
    # 2. do some data augmentation
    transforms.RandomHorizontalFlip(p=0.25),  # flip horizontal 25% of the time
    transforms.RandomRotation(degrees=30),  # test_acc = 0.6710
    # transforms.RandomAffine(degrees=15),      # test_acc = 0.6347
    # 3. convert to Tensor so PyTorch can use the data
    transforms.ToTensor()  # ultimate goal is to convert to Tensors
])


# re-create our CNN model
# needs an input image size of 64x64 pixels
# and also the batch size [B, C, H, W] -> Batch, Channel, Height, Width
# INPUT_SHAPE = 3  # RGB Channels
# NODES = 64
# OUTPUT_SHAPE = 3 (len of class_names)
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(TinyVGG, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 8 * 8,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))


# create test_step() function
def eval_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # put model in eval mode.
    model.eval()

    # setup loss and acc counters
    test_loss, test_acc = 0, 0

    # put in inference mode
    with torch.inference_mode():
        # loop through dataloader in batches
        for batch, (X, y) in enumerate(dataloader):
            # send values to target device
            X, y = X.to(device), y.to(device)

            # forward pass
            test_pred_logits = model(X)

            # calc and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # calc and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    # get average of each batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


# instantiate model
model = TinyVGG(input_shape=3,  # number of color channels RGB=3
                hidden_units=NODES,
                output_shape=len(class_names)
                ).to(device)

# load model from disk
model.load_state_dict(torch.load(f=MODEL_PATH_NAME))


start_time = timer()
unknown_img = cv2.imread(filepathname)
# openCV uses BGR so we need to convert to RGB
unknown_img_rgb = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2RGB)
# print(f"unknown image shape is: {unknown_img.shape}")   # height, width, channel

# cv2.imshow('Preview', unknown_img)
# cv2.waitKey()

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

# resize image to 64x64, same size expected by our model
unknown_img_rgb = cv2.resize(unknown_img_rgb, dsize=(64,64))
unknown_img_t = img_transform(unknown_img_rgb)  # convert to tensor
unknown_img_t = unknown_img_t.unsqueeze(dim=0)  # add batch size

# print(f"transformed image shape is: {unknown_img_t.shape}")


model.eval()
with torch.inference_mode():
    predicted_class_logits = model(unknown_img_t)
    print(f"filepathname: {filepathname}")
    print(f"\nlogits: {predicted_class_logits}")
    print(f"predicted object is a: {class_names[torch.argmax(predicted_class_logits)]}")
    cv2.putText(unknown_img, class_names[torch.argmax(predicted_class_logits)], (50,50),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3 )
    cv2.imshow(class_names[torch.argmax(predicted_class_logits)], unknown_img)
    end_time = timer()
    print(f"total inference time is: {end_time - start_time} seconds")

cv2.waitKey()



