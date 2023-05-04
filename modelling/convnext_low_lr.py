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

# %%
def convnext_tiny_():
    model = models.convnext_tiny(weights=True,)
    model.classifier[2] =nn.Linear(in_features=768, out_features=8, bias=True)
    return model

#%%
model_name = "convnext_tiny_lr_8e-5"
pretrained_model = "convnext"
#%%
convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)
convnext_transformer

# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_)
convnext.train_model(model_name, pretrained_model, num_epochs=3, cross_validation=True,test_model=False,batchsize_train_data=128,lr=8e-5)

#%%
model_name = "convnext_tiny_lr_1e-5"
pretrained_model = "convnext"
# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_)
convnext.train_model(model_name, pretrained_model, nnum_epochs=3, cross_validation=True,test_model=False,batchsize_train_data=128,lr=1e-5)

#%%
model_name = "convnext_tiny_lr_1e-6"
pretrained_model = "convnext"
# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_)
convnext.train_model(model_name, pretrained_model, num_epochs=3, cross_validation=True,test_model=False,batchsize_train_data=128,lr=1e-6)

