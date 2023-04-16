# %%
import os
print(os.getcwd())
if os.getcwd().endswith("modelling"):
    os.chdir("..")

#%%
is_cscs_run = True
if is_cscs_run:
    os.chdir("ccv1/tierli_ahluege/")
    print(os.getcwd())
#%%
from src.modelling import CCV1_Trainer
from src.augmentation import CCV1Transformer,None_Transform
from src.data_modules import DataModule
from torch import nn
import torch
from torchvision import transforms,models

#%%
# wandb Notebook Setup
%env "WANDB_NOTEBOOK_NAME" "ccv1_efficient_net"
%env WANDB_SILENT=True
# %%
# https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0
efficientnet_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", "efficientnet"
).getCompose()
# %%
def efficient_():
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
    nn.Linear(1280, 1000),  # dense layer takes a 2048-dim input and outputs 100-dim
    nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
    nn.Dropout(0.1),  # common technique to mitigate overfitting
    nn.Linear(
        1000, 8
    )) # final dense layer outputs 8-dim corresponding to our target classes
    return model
# %%
efficientnet = CCV1_Trainer(DataModule(efficientnet_transformer), efficient_)
efficientnet.train_model("without augmentation3", "efficientnet", num_epochs=6, test_model=False,batchsize_train_data=128)

# %%
efficientnet.submission("test_efficient")

