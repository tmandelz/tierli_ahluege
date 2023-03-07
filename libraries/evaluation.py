# %%
import torch
import pytorch_lightning as pl
import numpy as np
import wandb

# %%
class Evaluation():
    def __init__(self,data_classes:list) -> None:
        """
        Jan
        :param list data_classes: list of true labels (convert from int to str)
        """
        self.classes = data_classes

    def per_epoch(self,
                  loss_train:float,
                  pred_train:np.array,
                  label_train:np.array,
                  loss_val:float,
                  pred_val:np.array,
                  label_val:np.array
                  ) -> None:
        """
        Jan
        wandb log of different scores
        :param float loss_train: log loss of the training 
        :param np.array pred_train: prediction of the training
        :param np.array label_train: labels of the training
        :param float loss_val: log loss of the validation
        :param np.array pred_val: prediction of the validation
        :param np.array label_val: labels of the validation
        """
        wandb.log({"f1 train":self.f1_score(pred_train,label_train),
                   "Loss train":loss_train,
                   "Accuracy test":self.f1_score(pred_val,label_val),
                   "Loss test": loss_val})

    def per_model(self,label_val,pred_val) -> None:
        """
        Jan
        wandb log of a confusion matrix
        :param np.array pred_val: prediction of the validation
        :param np.array label_val: labels of the validation
        """ 
        wandb.log({"confusion matrix":wandb.sklearn.plot_confusion_matrix(label_val,pred_val,self.classes)})

    @staticmethod
    def f1_score(self,pred:np.array,label:np.array) -> float:
        """
        Jan
        f1 score of a given prediction and label
        :param np.array pred: prediction
        :param np.array label: 
        """
        pass