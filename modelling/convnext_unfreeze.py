
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
%env "WANDB_NOTEBOOK_NAME" "ccv1_convnext_unfreeze"
%env WANDB_SILENT=True

#%%
model = models.convnext_tiny(weights=True)
def convnext_tiny_unfreeze():
    model = models.convnext_tiny(weights=True)
    # Disable gradients on all model parameters to freeze the weights
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[2] =nn.Sequential(
        nn.Dropout(p=0.5),
    nn.Linear(in_features=768, out_features=128, bias=True),
    nn.Linear(in_features=128, out_features=8, bias=True))
    return model

# [i.shape for i in convnext_tiny_unfreeze().parameters()]
#%%
model_name = "convnext_tiny_freeze_weights_dropout"
pretrained_model = "convnext"
convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)

# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_unfreeze,)
convnext.train_model(model_name, pretrained_model, num_epochs=5, cross_validation=True, test_model=False, batchsize_train_data=128, lr = 3e-4,num_workers=0)

# %%
convnext.submission(model_name)
# %%
def convnext_small_unfreeze():
    model = models.convnext_small(weights=True)
    # Disable gradients on all model parameters to freeze the weights
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[2] =nn.Sequential(
        nn.Dropout(p=0.5),
    nn.Linear(in_features=768, out_features=128, bias=True),
    nn.Linear(in_features=128, out_features=8, bias=True))
    return model
# %%
model_name = "convnext_small_freeze_weights_dropout"
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_small_unfreeze,)
convnext.train_model(model_name, pretrained_model, num_epochs=5, cross_validation=True, test_model=False, batchsize_train_data=128, lr = 3e-4,num_workers=0)
# %%

convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)
def convnext_base_unfreeze_less():
    model = models.convnext_base(weights=True)
    # Disable gradients on all model parameters to freeze the weights
    n = len(list(model.parameters()))
    for param in model.parameters():
        n-=1
        if n < 5 + 2 * 8:
            break
        param.requires_grad = False
    model.classifier[2] =nn.Sequential(
    nn.Linear(in_features=1024, out_features=8, bias=True))
    return model

# %%
model_name = "convnext_base_freeze_less2"
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_base_unfreeze_less,)
convnext.train_model(model_name, pretrained_model, num_epochs=3, cross_validation=True, test_model=False, batchsize_train_data=128, lr = 4e-4,num_workers=0)
# %%
