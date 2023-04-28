
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
%env "WANDB_NOTEBOOK_NAME" "ccv1_convnext_unfreeze"
%env WANDB_SILENT=True

#%%
model = models.convnext_tiny(weights=True)
def convnext_tiny_unfreeze():
    model = models.convnext_tiny(weights=True)
    # Disable gradients on all model parameters to freeze the weights
    for param in model.parameters():
        param.requires_grad = True
    model.classifier[2] =nn.Linear(in_features=768, out_features=8, bias=True)
    return model
#%%

# model = convnext_tiny_unfreeze()
# for child in model.children():
#     for param in child.parameters():
#         param.requires_grad = True



# [i.shape for i in convnext_tiny_unfreeze().parameters()]
#%%
model_name = "convnext_tiny_unfreeze_weights"
pretrained_model = "convnext"
convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)

# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_unfreeze,)
convnext.train_model(model_name, pretrained_model, num_epochs=1, cross_validation=False, test_model=True, batchsize_train_data=128, lr = 3e-4)

# %%
convnext.submission(model_name)

# %%
