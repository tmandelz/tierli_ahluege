#%%
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
%env "WANDB_NOTEBOOK_NAME" "CCV1_resnet"
%env WANDB_SILENT=True


# %%
def resnet50_():
    model = models.resnet50(weights=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(
            100, 8
        ),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    return model
# %%
resnet_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", "resnet"
).getCompose()
# %%
resnet = CCV1_Trainer(DataModule(resnet_transformer), resnet50_)
resnet.train_model("resnet without augmentation", "resnet", num_epochs=6, test_model=False,batchsize_train_data=128)
# %%



resnet_transformer = CCV1Transformer(
    transforms.Compose([transforms.RandomRotation(20)]), "model_specific", "resnet"
).getCompose()
resnet = CCV1_Trainer(DataModule(resnet_transformer), resnet50_)
resnet.train_model("resnet with rotation (20)", "resnet", num_epochs=6, cross_validation=False,batchsize_train_data=128)
# %%
resnet_transformer = CCV1Transformer(
    transforms.Compose([transforms.RandomHorizontalFlip()]), "model_specific", "resnet"
).getCompose()
resnet = CCV1_Trainer(DataModule(resnet_transformer), resnet50_)
resnet.train_model("resnet with horizontal flip", "resnet", num_epochs=6, cross_validation=False,batchsize_train_data=128)
# %%

resnet_transformer = CCV1Transformer(
    transforms.Compose([transforms.RandomPerspective()]), "model_specific", "resnet"
).getCompose()
resnet = CCV1_Trainer(DataModule(resnet_transformer), resnet50_)
resnet.train_model("resnet with random perspective", "resnet", num_epochs=6, cross_validation=False,batchsize_train_data=128)
# %%
# resnet_transformer = CCV1Transformer(
#     transforms.Compose([transforms.AugMix()]), "model_specific", "resnet"
# ).getCompose()
# resnet = CCV1_Trainer(DataModule(resnet_transformer), resnet50_)
# resnet.train_model("resnet with augmix", "resnet", num_epochs=6, cross_validation=False,batchsize_train_data=128)
# %%
resnet_transformer = CCV1Transformer(
    transforms.Compose([transforms.ColorJitter(brightness=0.05)]), "model_specific", "resnet"
).getCompose()
resnet = CCV1_Trainer(DataModule(resnet_transformer), resnet50_)
resnet.train_model("resnet with color brightness 0.05", "resnet", num_epochs=6, cross_validation=False,batchsize_train_data=128)
# %%