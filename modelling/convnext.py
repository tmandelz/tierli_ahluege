#%% [markdown]
"""
Sources:
https://github.com/facebookresearch/ConvNeXt

@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
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
%env "WANDB_NOTEBOOK_NAME" "ccv1_convnext"
%env WANDB_SILENT=True

#%%
model = models.convnext_tiny(weights=True)
# %%
def convnext_tiny_():
    model = models.convnext_tiny(weights=True,)
    model.classifier[2] =nn.Linear(in_features=768, out_features=8, bias=True)
    return model


#%%
model_name = "convnext_tiny_val_sec1_2"
pretrained_model = "convnext"
convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)

# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=1, cross_validation=True,test_model=False,batchsize_train_data=128,lr = 3e-4,num_workers=0,decrease_security_validation=1.2)
# %%
model_name = "convnext_tiny"
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=1, cross_validation=True,test_model=False,batchsize_train_data=128,lr = 3e-4,num_workers=0)
# %%
convnext.submission(model_name)

# %%
model_name = "convnext_tiny_overfit_try"
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=20, cross_validation=False,test_model=False,batchsize_train_data=128,lr = 3e-4)

# %%
convnext.submission(model_name)

# %%
model_name = "convnext_tiny_weights_training"
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=20, cross_validation=False,test_model=False,batchsize_train_data=128,lr = 3e-4)

# %%
convnext.submission(model_name)

#%%
model_name = "convnext_tiny_cv"
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=5, cross_validation=True,test_model=False,batchsize_train_data=128,lr = 3e-4)

#%%

model_name = "convnext_tiny_megadetector"
convnext = CCV1_Trainer(DataModule(convnext_transformer,delete_unrecognized_mega=True,include_megadetector_train=True,include_megadetector_test=True), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=5, cross_validation=True,test_model=False,batchsize_train_data=128,lr = 3e-4,num_workers=0)
# %%
convnext.submission(model_name)

# %%
model_name = "convnext_tiny_exclude_megadetector"
convnext = CCV1_Trainer(DataModule(convnext_transformer,delete_recognized_mega=True), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=1, cross_validation=False,test_model=False,batchsize_train_data=128,lr = 3e-4,num_workers=0)
# %%
convnext.submission(model_name)

# %%
model_name = "convnext_tiny_megadetector_only_crop"
convnext = CCV1_Trainer(DataModule(convnext_transformer,delete_unrecognized_mega=False,include_megadetector_train=True,include_megadetector_test=True), convnext_tiny_,)
convnext.train_model(model_name, pretrained_model, num_epochs=3, cross_validation=True,test_model=False,batchsize_train_data=128,lr = 3e-4,num_workers=0)
#%%
def convnext_():
    model = models.convnext_tiny(weights=True,)
    model.classifier[2] =nn.Dropout(0.2)
    model.classifier.add_module("3",nn.Linear(in_features=768, out_features=8, bias=True))
    return model

#%%
model_name = "convnext_tiny_dropout_0.2"
pretrained_model = "convnext"
convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)
# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_,)
convnext.train_model(model_name, pretrained_model, num_epochs=5, cross_validation=False,test_model=False,batchsize_train_data=128,lr = 3e-4)

# %%
convnext.submission(model_name)
# %%
convnext._save_model("convnext_tiny_")

#%%
def convnext_small_():
    model = models.convnext_small(weights=True,)
    model.classifier[2] =nn.Linear(in_features=768, out_features=8, bias=True)
    return model
#%%
model_name = "convnext_small"
pretrained_model = "convnext"
convnext_transformer = CCV1Transformer(
    transforms.Compose([None_Transform()]), "model_specific", pretrained_model
).getCompose(True)
# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_small_,)
convnext.train_model(model_name, pretrained_model, num_epochs=5, cross_validation=False,test_model=False,batchsize_train_data=32,lr = 3e-4)

# %%
convnext.submission(model_name)


#%%
def convnext_():
    model = models.convnext_tiny(weights=True,)
    model.classifier[2] =nn.Linear(in_features=768, out_features=124, bias=True)
    model.classifier.add_module("3",nn.ReLU(inplace=True))
    model.classifier.add_module("4",nn.Dropout(0.5))
    model.classifier.add_module("5",nn.Linear(in_features=124, out_features=8, bias=True))
    return model

#%%
model_name = "convnext_tiny_augmix_dropout_0.5"
pretrained_model = "convnext"
#%%
convnext_transformer = CCV1Transformer(
    transforms.Compose([transforms.AugMix()]), "model_specific", pretrained_model
).getCompose(True)

# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_)
convnext.train_model(model_name, pretrained_model, num_epochs=8, cross_validation=False,test_model=False,batchsize_train_data=128)

# %%
convnext.submission(model_name)

#%%
model_name = "convnext_tiny_horizontalflip_dropout_0.5"
pretrained_model = "convnext"
#%%
convnext_transformer = CCV1Transformer(
    transforms.Compose([transforms.RandomHorizontalFlip()]), "model_specific", pretrained_model
).getCompose(True)
convnext_transformer

# %%
convnext = CCV1_Trainer(DataModule(convnext_transformer), convnext_)
convnext.train_model(model_name, pretrained_model, num_epochs=8, cross_validation=False,test_model=False,batchsize_train_data=128)

# %%
convnext.submission(model_name)
# %%
