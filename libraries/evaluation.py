# %%
import torch
import pytorch_lightning as pl
import numpy as np
import wandb

# %%
class Evaluation():
    def __init__(self,device,data_classes) -> None:
        self.device = device
        self.classes = data_classes
    def per_batch(self) -> None:
        pass
    def per_epoch(self) -> None:
        wandb.log({"Accuracy train":accuracy.mean(),"Accuracy test": self.f1_score(pred_test,label_test),"Train loss":loss_train.mean(),"Test loss": loss_test})

    def per_model(self) -> None:
        wandb.log({"confusion matrix":wandb.sklearn.plot_confusion_matrix(label_test,pred_test,classes)})
    
    @staticmethod
    def f1_score(self) -> None:
        pass