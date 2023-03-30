#%%
import os
print(os.getcwd())
if os.getcwd().endswith("modelling"):
    os.chdir("..")

#%%
is_cscs_run = False
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
%env "WANDB_NOTEBOOK_NAME" "ccv1_convnext"
%env WANDB_SILENT=True
#%%
def convnext_():
    model = models.convnext_base(weights=True)
    model.classifier=model.fc = nn.Sequential(
            nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
            nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
            nn.Dropout(0.1),  # common technique to mitigate overfitting
            nn.Linear(
                100, 8
            ),  # final dense layer outputs 8-dim corresponding to our target classes
        )
    return model
convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", "resnet"
).getCompose()
# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_)
convnext.train_model("convnext without augmentation", "convnext", num_epochs=6, test_model=False,batchsize_train_data=128)
# %%
