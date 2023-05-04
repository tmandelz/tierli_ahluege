#%% [markdown]
"""

"""
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
%env "WANDB_NOTEBOOK_NAME" "ccv1_additional_models"
%env WANDB_SILENT=True
# %%
def swin_tiny_():
    model = models.swin_t(weights="IMAGENET1K_V1")
    model.head =nn.Linear(in_features=768, out_features=8, bias=True)
    return model


#%%
model_name = "swin_tiny"
pretrained_model = "swin"
swin_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)

# %%
swin = CCV1_Trainer(DataModule(swin_transformer), swin_tiny_,)
swin.train_model(model_name, pretrained_model, num_epochs=2, cross_validation=True,test_model=False,batchsize_train_data=128,lr = 3e-4,num_workers=0)
# %%


def inceptionv3_():
    model = models.inception_v3(weights="IMAGENET1K_V1")
    model.aux_logits=False
    model.fc =nn.Linear(in_features=2048, out_features=8, bias=True)
    return model


#%%
model_name = "inception_base"
pretrained_model = "inceptionv3"
inception_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)

# %%
inception = CCV1_Trainer(DataModule(inception_transformer), inceptionv3_,)
inception.train_model(model_name, pretrained_model, num_epochs=2, cross_validation=True,test_model=False,batchsize_train_data=128,lr = 3e-4,num_workers=0)
# %%
