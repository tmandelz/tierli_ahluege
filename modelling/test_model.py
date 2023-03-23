# %%
import os

if os.getcwd().endswith("modelling"):
    os.chdir("..")

import pandas as pd
import numpy as np
from src.modelling import del_model
from src.augmentation import CCV1Transformer,None_Transform
from src.data_modules import DataModule,ImagesDataset
import torch
from torch import nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms


# %%
# only for testing
train_features = pd.read_csv("./competition_data/train_features.csv", index_col="id")
train_labels = pd.read_csv("./competition_data/train_labels.csv", index_col="id")
train_features.sample(160,random_state=1).to_csv("./competition_data/val_features.csv")
train_labels.sample(160,random_state=1).to_csv("./competition_data/val_labels.csv")
# %%



test_transformer = CCV1Transformer(
    transforms.Compose([transforms.ColorJitter(brightness=0.1)]), "standard", "mlp"
).getCompose()

test_transformer

#%%
class base_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(150528, 8)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x

model = base_mlp()
simple_mlp = del_model(
    DataModule(basic_transform=test_transformer),
    model,
    batchsize_train_data=16,
)
optimizer = optim.Adam(model.parameters())
# %%
simple_mlp.train_model(optimizer, "base_mlp", "overfitting", 10, test_model=True)
#%%
simple_mlp.submit_file("test")

# %%
# https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0
efficientnet_transformer = CCV1Transformer(
    transforms.Compose([transforms.RandomRotation(degrees=(-90,90))]), "standard", "efficientnet"
).getCompose()
efficientnet_transformer

#%%
import wandb
wandb.finish()

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
efficientnet = del_model(DataModule(efficientnet_transformer), efficient)
efficientnet.train_model(optimizer, "efficientnet", "testrun1", 1,test_model=True)
# %%
efficientnet.submit_file("test_efficient")
#%%