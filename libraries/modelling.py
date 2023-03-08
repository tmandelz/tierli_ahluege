# %%
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from libraries.data_modules import DataModule
from libraries.evaluation import Evaluation

# %%
def set_seed(seed:int=42):
    """
    Jan
    Function to set the seed for the gpu and the cpu
    :param int seed: DON'T CHANGE
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class del_model:
    def __init__(self,data_model:DataModule,model:nn.Module,device:torch.device=device,batchsize_train_data:int=64) -> None:
        """
        Jan
        Load the train/test/val data.
        :param DataModule data_model: instance where the 3 dataloader are available.
        :param nn.Module model: pytorch deep learning module
        :param torch.device device: used device for training
        :param int batchsize: batchsize of the training data
        """
        self.train_loader = data_model.train_dataloader(batchsize_train_data)
        self.val_loader = data_model.val_dataloader()
        self.test_loader = data_model.test_dataloader()
        self.model = model
        self.device = device
        self.evaluation = Evaluation()
    def train_model(self,
                    optimizer:torch.optim,
                    modeltyp:str,
                    run_name:str,
                    num_epochs:int,
                    loss_module:nn=nn.CrossEntropyLoss(),
                    test_model:bool=False
                    ):
        """
        Jan
        To train a pytorch model.
        :param torch.optim optimizer: optimizer for the training
        :param str modeltyp: Modeltyp (architectur) of the model -> to structure wandb
        :param str run_name: Name of a single run.
        :param int num_epochs:
        :param nn.CrossEntropyLoss loss_module: Loss used for the competition
        :param int test_model: If true, it only loops over the first train batch. -> For the overfitting test.
        """ 
        # wandb setup
        set_seed()
        run = wandb.init(
            project="competition",
            entity="deeptier",
            name=run_name,
            config={
            "learning_rate":optimizer.defaults["lr"],
            "epochs":num_epochs,
            "Modelltyp":modeltyp,
            }
            )
        # training
        self.model.train()
        self.model.to(device)
        if test_model:
            self.train_loader = [next(iter(self.train_loader))]
        for epoch in tqdm(range(num_epochs)):
            loss_train = np.array([])
            label_train_data = np.empty((0, 8))
            pred_train_data = np.array([])
            for batch in self.train_loader:
                
                # calc gradient
                data_inputs = batch["image"].to(device)
                data_labels = batch["label"].to(device)

                preds = self.model(data_inputs)
                loss = loss_module(preds, data_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.evaluation.per_batch(loss)
                # data for evaluation
                label_train_data = np.concatenate((label_train_data,data_labels.data.cpu().numpy()),axis=0)
                predict_train = torch.argmax(preds, 1).data.cpu().numpy()
                pred_train_data = np.concatenate((pred_train_data,predict_train),axis=0)
                loss_train = np.append(loss_train,loss.item())

            # wandb per epoch
            pred_val,label_val = self.predict(self.model,self.val_loader)
            loss_val = loss_module(torch.tensor(pred_val), torch.tensor(label_val))
            self.evaluation.per_epoch(loss_train.mean(),pred_train_data,label_train_data,loss_val,np.argmax(pred_val, axis=1),label_val)
                    
        # wandb per run
        self.evaluation.per_model(label_val,pred_val)

        # prediction off the test set
        self.prediction_test,_ = self.predict(self.model,self.test_loader)

    def predict(self,model:nn.Module,data_loader:DataLoader):
        """
        Jan
        Prediction for a given model and dataset
        :param nn.Module model: pytorch deep learning module
        :param DataLoader data_loader: data for a prediction
        """
        model.eval()
        predictions = np.empty((0, 8))
        true_labels = np.empty((0, 8))
        with torch.no_grad(): # Deactivate gradients for the following code
            for batch in data_loader:

                # Determine prediction of model
                data_inputs = batch["image"].to(self.device)
                
                preds = model(data_inputs)
                preds = torch.sigmoid(preds)
                predictions = np.concatenate((predictions,preds.data.cpu().numpy()),axis=0)
                if len(batch) == 3:                    
                    data_labels = batch["label"].to(self.device)
                    true_labels = np.concatenate((true_labels,data_labels.data.cpu().numpy()),axis=0)
        model.train()
        return predictions,true_labels