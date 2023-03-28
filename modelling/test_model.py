# %%
import os
print(os.getcwd())
if os.getcwd().endswith("modelling"):
    os.chdir("..")

#%%
is_cscs_run = False
if is_cscs_run:
    os.chdir("ccv1/tierli_ahluege")
    print(os.getcwd())
#%%

import pandas as pd
import numpy as np
from src.modelling import CCV1_Trainer
from src.augmentation import CCV1Transformer,None_Transform
from src.data_modules import DataModule,ImagesDataset
from torch import nn
import torch
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms

#%%
# wandb Notebook Setup
%env "WANDB_NOTEBOOK_NAME" "SGDS_DEL_MC1_Notebook"
%env WANDB_SILENT=True

# %%
test_transformer = CCV1Transformer(
    None_Transform(), "standard", "mlp"
).getCompose()

test_transformer

# %%
class base_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(150528, 8)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x

# model = base_mlp()
simple_mlp = CCV1_Trainer(
    DataModule(basic_transform=test_transformer),
    base_mlp,
)
# optimizer = optim.Adam(model.parameters())
# %%
simple_mlp.train_model("overfitting","base_mlp", batchsize_train_data=16, num_epochs=6,test_model=True,lr=0.00005)
#%%
simple_mlp.submit_file("test")

# %%
# https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0
efficientnet_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "standard", "efficientnet"
).getCompose()
efficientnet_transformer

# # %%
# def efficient_():
#     model = models.efficientnet_b0(pretrained=True)
#     model.classifier = nn.Sequential(
#     nn.Linear(1280, 1000),  # dense layer takes a 2048-dim input and outputs 100-dim
#     nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
#     nn.Dropout(0.1),  # common technique to mitigate overfitting
#     nn.Linear(
#         1000, 8
#     )) # final dense layer outputs 8-dim corresponding to our target classes
#     return model
# # %%
# efficientnet = del_model(data_modules.DataModule(), efficient_)
# efficientnet.train_model("efficientnet", "testrun1", 10, test_model=True,batchsize_train_data=16)

# # %%
# efficientnet.submit_file("test_efficient")
#%%