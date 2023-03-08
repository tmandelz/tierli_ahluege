# %%
import os
if os.getcwd().endswith('model'):
    os.chdir('..')

import pandas as pd
import numpy as np
from libraries.modelling import del_model
from libraries import data_modules
import torch
from torch import nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
# %%
# only for testing
# train_features = pd.read_csv("../competition_data/train_features.csv", index_col="id")
# train_labels = pd.read_csv("../competition_data/train_labels.csv", index_col="id")
# train_features.sample(160,random_state=1).to_csv("../competition_data/val_features.csv")
# train_labels.sample(160,random_state=1).to_csv("../competition_data/val_labels.csv")
# %%

test_transformer = transforms.Compose([
                transforms.Resize((20, 20)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
#%%
class base_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1200, 8)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

model = base_mlp()
simple_mlp = del_model(data_modules.DataModule(basic_transform=test_transformer),model,batchsize_train_data=16)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# %%
simple_mlp.train_model(optimizer,"base_mlp","testrun1",5,test_model=True)

# %%
model = models.efficientnet_b2(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(150528, 1000),  # dense layer takes a 2048-dim input and outputs 100-dim
    nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
    nn.Dropout(0.1),  # common technique to mitigate overfitting
    nn.Linear(1000, 8),  # final dense layer outputs 8-dim corresponding to our target classes
)

efficientnet = del_model(data_modules.DataModule(),model)
efficientnet.train_model(optimizer,"resnet50","testrun1",1)
# %%
test = data_modules.DataModule(basic_transform=test_transformer).val.data
# %%
test[test["site"] == test["site"].value_counts().index[0]]
# %%
test.iloc[[1,2]]
# %%
np.random.choice(range(7),size=16,replace=False)
# %%
a = {'a':1, 'b':2, 'c':3}
b = {'d':1, 'e':2, 'f':3}
c = {1:1, 2:2, 3:3}
merge = {**a, **b, **c}
print(merge)
# %%
