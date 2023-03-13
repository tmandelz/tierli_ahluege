# %%
import os

if os.getcwd().endswith("model"):
    os.chdir("..")

import pandas as pd
import numpy as np
from libraries.modelling import del_model
from libraries import data_modules
import torch
from torch import nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms

# %%
# only for testing
# train_features = pd.read_csv("../competition_data/train_features.csv", index_col="id")
# train_labels = pd.read_csv("../competition_data/train_labels.csv", index_col="id")
# train_features.sample(160,random_state=1).to_csv("../competition_data/val_features.csv")
# train_labels.sample(160,random_state=1).to_csv("../competition_data/val_labels.csv")
# %%

test_transformer = transforms.Compose(
    [
        transforms.Resize((20, 20)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)
#%%
class base_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1200, 8)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x


model = base_mlp()
simple_mlp = del_model(
    data_modules.DataModule(basic_transform=test_transformer),
    model,
    batchsize_train_data=16,
)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# %%
simple_mlp.train_model(optimizer, "base_mlp", "testrun1", 5, test_model=True)
# simple_mlp.submit_file("test")

# %%
# https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0
efficientnet_transfromer = transforms.Compose(
    [
        transforms.Resize(
            (256, 256), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
efficient = models.efficientnet_b0(pretrained=True)
efficient.classifier = nn.Sequential(
    nn.Linear(1280, 1000),  # dense layer takes a 2048-dim input and outputs 100-dim
    nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
    nn.Dropout(0.1),  # common technique to mitigate overfitting
    nn.Linear(
        1000, 8
    ),  # final dense layer outputs 8-dim corresponding to our target classes
)
optimizer = optim.SGD(efficient.parameters(), lr=0.1, momentum=0.9)
efficientnet = del_model(data_modules.DataModule(), efficient)
efficientnet.train_model(optimizer, "efficientnet", "testrun1", 1)
# %%
