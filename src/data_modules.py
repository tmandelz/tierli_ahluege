# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import ast


# %%
class ImagesDataset(Dataset):
    """
    Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(
        self, x_df: pd.DataFrame, transform: transforms, y_df: pd.DataFrame = None, include_megadetector: bool = False, threshhold_megadetector: float = 0.5
    ):
        """
        :param pd.DataFrame x_df: links of the jpg
        :param transforms transform: for basic transformation (like normalisation)
        :param pd.DataFrame y_df: labels
        :param bool include_megadetector: add a megadetector transformation
        :param float threshhold_megadetector: threshhold for box if the megadetector is activated
        """
        self.data = x_df
        self.label = y_df
        self.transform = transform
        self.include_megadetector = include_megadetector
        self.threshhold_megadetector = threshhold_megadetector

    def __getitem__(self, index: int) -> dict:
        """
        :param int index: index of the data path

        :return: dictionary of id,image,label
        :rtype: dict

        """
        # get image from path
        path = r"./competition_data/" + self.data.iloc[index]["filepath"]
        image = Image.open(path).convert("RGB")

        # crop images with bounding boxes of megadetector
        if self.include_megadetector and self.data.iloc[index]["conf"] > self.threshhold_megadetector:
            # get bounding box by coordinates and shapes
            y, x, height, width = ast.literal_eval(
                self.data.iloc[index]["bbox"])
            image_tensor = transforms.ToTensor()(image)
            image = transforms.ToPILImage()(
                transforms.functional.crop(image_tensor, y, x, height, width))

        # transform the picture
        image = self.transform(image)
        # get image id from index
        image_id = self.data.index[index]

        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(
                self.label.iloc[index].values, dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        basic_transform: transforms,
        train_features_path: str = "./competition_data/trainfeatures_megadet_bbox_split.csv",
        train_labels_path: str = "./competition_data/train_labels_with_split.csv",
        test_features_path: str = "./competition_data/testfeatures_megadet_bbox.csv",
        include_megadetector_train: bool = False,
        include_megadetector_test: bool = False,
        threshhold_megadetector:float=0.5,
        max_threshhold_megadetector:float = 1.01,
        delete_unrecognized_mega=False,
        delete_recognized_mega=False
    ) -> None:
        """
        Jan
        :param transforms basic_transform: basic tranformation -> default resize(224,224), ToTensor, standardize
        :param str train_features_path:
        :param str train_labels_path:
        :param str test_features_path:
        :param bool include_megadetector_train: add a megadetector transformation for training
        :param bool include_megadetector_test: add a megadetector transformation for testing
        :param float threshhold_megadetector: threshhold for box if the megadetector is activated
        :param float max_threshhold_megadetector: maximal threshhold for box if the megadetector is activated
        :param delete_unrecognized_mega: delete image where the megadetector don't recognize images
        :param delete_recognized_mega: delete image where the megadetector recognize images
        """
        # load_data
        self.train_features = pd.read_csv(train_features_path, index_col="id")
        self.train_labels = pd.read_csv(train_labels_path, index_col="id")
        test_features = pd.read_csv(test_features_path, index_col="id")

        # activation for megadetector
        self.include_megadetector_train = include_megadetector_train
        self.include_megadetector_test = include_megadetector_test
        self.threshhold_megadetector = threshhold_megadetector
        self.max_threshhold_megadetector = max_threshhold_megadetector
        self.delete_unrecognized_mega = delete_unrecognized_mega
        self.delete_recognized_mega = delete_recognized_mega

        if delete_recognized_mega and delete_unrecognized_mega:
            print("You deleted all the Data")
            raise ValueError

        if self.include_megadetector_test and self.delete_unrecognized_mega:
           test_features = test_features[test_features["conf"]>self.threshhold_megadetector]
           test_features = test_features[(test_features["conf"]>self.max_threshhold_megadetector)==False]

        if self.delete_recognized_mega:
            test_features = test_features[(
                test_features["conf"] > self.threshhold_megadetector) == False]

        # prepare transforms
        self.basic_transform = basic_transform

        # exclude the 2nd transformation in val und test set -> data augmentation only used by trainset
        self.exclude_augmentation_transformer = transforms.Compose(
            basic_transform.transforms[:1] + basic_transform.transforms[1 + 1:])

        # exclude data augmentation compose
        self.test = ImagesDataset(
            test_features, self.exclude_augmentation_transformer, include_megadetector=include_megadetector_test, threshhold_megadetector=threshhold_megadetector)

    def prepare_data(self,
                     fold_number) -> None:
        val_features = self.train_features.loc[self.train_features["split"]
                                               == fold_number, self.train_features.columns != "split"]
        train_features = self.train_features.loc[self.train_features["split"]
                                                 != fold_number, self.train_features.columns != "split"]

        val_labels = self.train_labels.loc[self.train_labels["split"]
                                           == fold_number, self.train_labels.columns != "split"]
        train_labels = self.train_labels.loc[self.train_labels["split"]
                                             != fold_number, self.train_labels.columns != "split"]
        # delete files for megadetector
        if self.include_megadetector_test and self.delete_unrecognized_mega:
            val_labels = val_labels[val_features["conf"]>self.threshhold_megadetector]
            val_labels = val_labels[(val_features["conf"]>self.max_threshhold_megadetector)==False]

            val_features = val_features[val_features["conf"]>self.threshhold_megadetector]
            val_features = val_features[(val_features["conf"]>self.max_threshhold_megadetector)==False]

        if self.include_megadetector_train and self.delete_unrecognized_mega:
            train_labels = train_labels[train_features["conf"]>self.threshhold_megadetector]
            train_labels = train_labels[(train_features["conf"]>self.max_threshhold_megadetector)==False]
            train_features = train_features[train_features["conf"]>self.threshhold_megadetector]
            train_features = train_features[(train_features["conf"]>self.max_threshhold_megadetector)==False]

        if self.delete_recognized_mega:
            val_labels = val_labels[(
                val_features["conf"] > self.threshhold_megadetector) == False]
            val_features = val_features[(
                val_features["conf"] > self.threshhold_megadetector) == False]
            train_labels = train_labels[(
                train_features["conf"] > self.threshhold_megadetector) == False]
            train_features = train_features[(
                train_features["conf"] > self.threshhold_megadetector) == False]

        self.train = ImagesDataset(
            train_features, self.basic_transform, train_labels, self.include_megadetector_train, self.threshhold_megadetector)
        self.val = ImagesDataset(
            val_features, self.exclude_augmentation_transformer, val_labels, self.include_megadetector_test, self.threshhold_megadetector)

    def train_dataloader(self, batch_size: int = 128, num_workers: int = 16):
        """
        :param int batch_size: batch size of the training data -> default 64
        :param int num_workers: number of workers for the data loader (optimize if GPU usage not optimal) -> default 16
        """
        return DataLoader(self.train, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=256)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=256)

# %%
