# %%
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
# %%
class ImagesDataset(Dataset):
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df:pd.DataFrame,transform:transforms,y_df:pd.DataFrame=None):
        self.data = x_df
        self.label = y_df
        self.transform = transform

    def __getitem__(self, index:int):
        image = Image.open("../competition_data/" + self.data.iloc[index]["filepath"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, 
                                 dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)

class DataModule(pl.LightningDataModule):  
    def __init__(self, 
                test_transform:transforms,
                train_transform:transforms,
                train_features_path="../competition_data/train_features.csv",
                test_features_path="../competition_data/test_features.csv",
                train_labels_path="../competition_data/train_labels.csv"):
        # load_data
        train_features = pd.read_csv(train_features_path, index_col="id")
        test_features = pd.read_csv(test_features_path, index_col="id")
        train_labels = pd.read_csv(train_labels_path, index_col="id")
        
        # train val split
        train_y,train_x,val_y,val_x = self.train_test_split(train_features,train_labels)

        # prepare transforms
        self.train = ImagesDataset(train_x,test_transform,train_y)
        self.val = ImagesDataset(val_x,test_transform,val_y)
        self.test = ImagesDataset(test_features,test_transform)
        
        #self.train, self.val = self.train_test_split(train_data)
    @staticmethod
    def train_test_split(features:pd.DataFrame,labels:pd.DataFrame):
        # TODO define split
        return random_split(features, [55000, 5000])
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64)