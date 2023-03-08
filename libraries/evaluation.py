# %%
import torch
import pytorch_lightning as pl
import numpy as np
import wandb
from sklearn.metrics import f1_score

# %%
classes = ["antelope_duiker","bird","blank","civet_genet","hog","leopard","monkey_prosimian","rodent"]
class Evaluation():
    def __init__(self,data_classes:list=classes) -> None:
        """
        Jan
        :param list data_classes: list of true labels (convert from int to str)
        """
        self.classes = data_classes

    def per_batch(self,loss_batch):
        wandb.log({"loss batch": loss_batch})

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
        
        wandb.log({"Loss train":loss_train,
                   "Loss val": loss_val})
        for animal in range(len(self.classes)):
            wandb.log({
                f"train f1_score von {self.classes[animal]}": f1_score(label_train[:,animal],pred_train==animal),
                f"validation f1_score von {self.classes[animal]}" : f1_score(label_val[:,animal],pred_val==animal)           
            })

        
    def per_model(self,label_val,pred_val) -> None:
        """
        Jan
        wandb log of a confusion matrix
        :param np.array pred_val: prediction of the validation
        :param np.array label_val: labels of the validation
        """
        wandb.log({"confusion matrix":wandb.sklearn.plot_confusion_matrix(np.argmax(label_val,axis=1),np.argmax(pred_val,axis=1),self.classes)})

    @staticmethod
    def f1_score(pred:np.array,label:np.array) -> float:
        """
        Jan
        f1 score of a given prediction and label
        :param np.array pred: prediction
        :param np.array label: 
        """
        pass