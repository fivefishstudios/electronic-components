# Learn about custom dataset (i.e. using our own images) for Machine Learning
# updated... Using Transfer Learning: EfficientNet base model adapter to our problem set
# 11/3/22

import os
import requests
import zipfile
from pathlib import Path
import torch
import torchvision
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

CPU_COUNT = os.cpu_count()
BATCH_SIZE = 32
EPOCHS = 50
# NODES = 64
LEARNING_RATE = 0.00045  # lower converges better/faster

print(f"Our PyTorch version is: {torch.__version__}")

# setup cuda if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"NOTE: All calculations will be done in {device}")

# setup path to our /data folder
data_path = Path("data")  # Posix format
image_path = data_path / Path("components")

# set up train and test path
train_dir = image_path / Path("train")
test_dir = image_path / Path("test")
print(f"{train_dir} and {test_dir} are our TRAIN and TEST directories")

# transforming data (i.e. our input source, images)
# create a transform function that we will use in DataLoader
data_transform = transforms.Compose([  # Compose is used to serialize our tranforms functions
    # transform operations are a list
    # 1. resize image to be smaller
    transforms.Resize(size=(224, 224)),
    # 2. do some data augmentation
    transforms.RandomHorizontalFlip(p=0.25),  # flip horizontal 25% of the time
    transforms.RandomRotation(degrees=30),
    # transforms.RandomAffine(degrees=15),
    # 3. convert to Tensor so PyTorch can use the data
    transforms.ToTensor(),  # ultimate goal is to convert to Tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# print(data_transform)
# Let's try our data_tansform() function to see if it's working
# grab an image
# image_sample = test_dir / "steak" / "100274.jpg"
# # open image sample
# with Image.open(image_sample) as img:
#     img.show()
#     # do some transform
#     img_transformed = data_transform(img)
#     # plt.imshow(img_transformed.permute(1, 2, 0))
#     cv2.imshow('Transformed', img_transformed.permute(1,2,0).numpy())
#     cv2.waitKey(0)

# Load our custom dataset
train_data = datasets.ImageFolder(
    root=train_dir,
    transform=data_transform,
    target_transform=None
)
# print(f"train_data: {train_data}")

test_data = datasets.ImageFolder(
    root=test_dir,
    transform=data_transform,
    target_transform=None
)
# print(f"\ntest_data: {test_data}")

# get class names from our dataset
class_names = train_data.classes
print(f"\nTRAIN class_names: {class_names}")
print(f"TEST class_names: {train_data.classes}")

# sample_img=train_data[10][0]
# print(train_data[0][0])
# print(f"Sample image is a {class_names[train_data[0][1]]}")

# sample_img = cv2.cvtColor(sample_img.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
# cv2.imshow(class_names[train_data[10][1]], sample_img)
# cv2.waitKey(0)

# turn our ImageFolder dataset into DataLoader so we can batch it and iterate through it


train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=CPU_COUNT,
    multiprocessing_context=get_context('loky')
)
# print(train_dataloader)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=CPU_COUNT,
    multiprocessing_context=get_context('loky')
)
# print(test_dataloader)

# check if dataloader is working, i.e. iterable
# img, label = next(iter(train_dataloader))
# print(f"\nimage shape is: {img.shape}")
# print(f"label is: {label}")


# create our CNN model
# INPUT_SHAPE = 3  # RGB Channels
# NODES = 64
# OUTPUT_SHAPE = 3 (len of class_names)
# class TinyVGG(nn.Module):
#     def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
#         super(TinyVGG, self).__init__()
#         self.conv_block_1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_shape,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2)
#         )
#         self.conv_block_2 = nn.Sequential(
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2)
#         )
#         self.conv_block_3 = nn.Sequential(
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hidden_units,
#                       out_channels=hidden_units,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=hidden_units * 16 * 16,
#                       out_features=output_shape)
#         )
#
#     def forward(self, x):
#         return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))
#

# TRANSFER LEARNING -> Setup the model with pretrained weights and send it to the target device (torchvision
# v0.13+)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT  # .DEFAULT = best available weights
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# partially freeze the model
for param in model.features.parameters():
    # print(param)
    param.requires_grad = False

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# override classifier method in model class
# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)


# check if our model works
# pred = model(img)
# print(f"prediction is: {torch.argmax(torch.softmax(pred, dim=1), dim=1)}")

# using torchinfo to get info about our model
# info_summary = summary(model, input_size=[1, 3, 64, 64])

# create modular train_step() and test_step() function
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # put in training mode our model
    model.train()

    # init training loss and accuracy
    train_loss, train_acc = 0, 0

    # loop through dataloader in batches
    for batch, (X, y) in enumerate(dataloader):
        # send to device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # calc loss and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss

        # optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        # optmizer step
        optimizer.step()

        # calc accuracy and accumulate
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # calc average per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


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


# create a training function
def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
                epochs: int = 5):
    # create results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # loop through training and test stps
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = eval_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        # print status
        print(f"{epoch}: "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f} | "
              )

        # update dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # return dictionary
    return results


# create training loop

# model = TinyVGG(input_shape=3,  # number of color channels RGB=3
#                 hidden_units=NODES,
#                 output_shape=len(train_data.classes)
#                 ).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

start_time = timer()
model_results = train_model(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=EPOCHS
)

end_time = timer()
print(f"\nTotal training time: {end_time - start_time} seconds")

# Save this model to disk for later recall
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_FILENAME = "electronic_componentsV2.pth"
MODEL_SAVEPATHNAME = MODEL_PATH / MODEL_FILENAME

# save model state dictionary
print(f"Saving model to {MODEL_SAVEPATHNAME}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVEPATHNAME)
