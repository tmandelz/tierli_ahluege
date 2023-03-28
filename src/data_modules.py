# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


# %%
class ImagesDataset(Dataset):
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(
        self, x_df: pd.DataFrame, transform: transforms, y_df: pd.DataFrame = None
    ):
        """
        :param pd.DataFrame x_df: links of the jpg
        :param transforms transform: for basic transformation (like normalisation)
        :param pd.DataFrame y_df: labels
        """
        self.data = x_df
        self.label = y_df
        self.transform = transform

    def __getitem__(self, index: int) -> dict:
        """
        :param int index: index of the data path

        :return: dictionary of id,image,label
        :rtype: dic

        """
        image = Image.open(
            "./competition_data/" + self.data.iloc[index]["filepath"]
        ).convert("RGB")
        
        image = self.transform(image)
        image_id = self.data.index[index]

        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        basic_transform: transforms,
        train_features_path: str = "./competition_data/train_features.csv",
        train_labels_path: str = "./competition_data/train_labels.csv",
        val_features_path: str = "./competition_data/val_features.csv",
        val_labels_path: str = "./competition_data/val_labels.csv",
        test_features_path: str = "./competition_data/test_features.csv",
    ):
        """
        Jan
        :param str train_features_path:
        :param str train_labels_path:
        :param str val_features_path:
        :param str val_labels_path:
        :param str test_features_path:
        :param transforms basic_transform: basic tranformation -> default resize(224,224), ToTensor, standardize
        """
        # load_data
        train_features = pd.read_csv(train_features_path, index_col="id")
        train_labels = pd.read_csv(train_labels_path, index_col="id")

        val_features = pd.read_csv(val_features_path, index_col="id")
        val_labels = pd.read_csv(val_labels_path, index_col="id")

        test_features = pd.read_csv(test_features_path, index_col="id")

        # prepare transforms
        self.train = ImagesDataset(train_features, basic_transform, train_labels)
        
        # exclude the 2nd transformation in val und test set -> data augmentation only used by trainset
        exclude_augmentation_transformer = transforms.Compose(basic_transform.transforms[:1] + basic_transform.transforms[1 + 1:])
        
        # exclude data augmentation compose
        self.val = ImagesDataset(val_features, exclude_augmentation_transformer, val_labels)
        # exclude data augmentation compose
        self.test = ImagesDataset(test_features, exclude_augmentation_transformer)

    def train_dataloader(self, batch_size: int = 64):
        """
        :param int batch_size: batch size of the training data -> default 64
        """
        return DataLoader(self.train, batch_size=batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64)

# %%
