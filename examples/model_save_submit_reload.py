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
%env "WANDB_NOTEBOOK_NAME" "SGDS_DEL_MC1_Notebook"
%env WANDB_SILENT=True

# %%
test_transformer = CCV1Transformer(
    None_Transform(), "overfitting", "mlp"
).getCompose()

# %%
class base_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1200, 8)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x

simple_mlp = CCV1_Trainer(
    DataModule(basic_transform=test_transformer),
    base_mlp,
)
#%%
simple_mlp.train_model("overfitting_test","base_mlp", batchsize_train_data=4,cross_validation=False, num_epochs=1,test_model=True,lr=0.1)

#%%
simple_mlp.submission("overfitting_Test_model")

#%%
reloaded_model = base_mlp()
reloaded_model.load_state_dict(torch.load(f"./model_submit/overfitting_Test_model.pth"))