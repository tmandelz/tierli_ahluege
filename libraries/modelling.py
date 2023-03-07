# %%
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader, random_split
from PIL import Image
import pytorch_lightning as pl
import numpy as np
import wandb
import tqdm

# %%

# %%
def set_seed():
    """
    Function to set the seed for the gpu and the cpu
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class del_model:
    def __init__(self) -> None:
        pass
    def train_model(self,model, optimizer, data_loader_train,data_loader_test, loss_module ,modeltyp, num_epochs=100,test_model = False,run_name="basic_test"):
        # wandb setup
        set_seed()
        run = wandb.init(
            project="Deep Learning MC1",
            entity = "jan-zwicky",
            name=run_name,
            config={
            "learning_rate": optimizer.defaults["lr"],
            "epochs": num_epochs,
            "Modelltyp": modeltyp,
            }
            )
        # training
        model.train()
        if test_model:
            data_loader_train = [next(iter(data_loader_train))]
        for epoch in tqdm(range(num_epochs)):
            accuracy = np.array([])
            loss_train = np.array([])
            for data_inputs, data_labels in data_loader_train:
                
                # calc gradient
                data_inputs = data_inputs.to(device)
                data_labels = data_labels.to(device)

                preds = model(data_inputs)
                loss = loss_module(preds, data_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # evaluation
                predict_train = torch.argmax(torch.sigmoid(preds), 1).data.cpu().numpy()
                label_train = data_labels.data.cpu().numpy()
                accuracy = np.concatenate((accuracy,np.array(predict_train==label_train)),axis=0)
                loss_train = np.append(loss_train,loss.item())

            # TODO evaluation class per model
            pred_test,label_test = predict(model,data_loader_test)
            loss_test = loss_module(torch.tensor(pred_test), torch.tensor(label_test))

                    
        # TODO evaluation class per model
    @staticmethod
    def predict(model, data_loader):
        model.eval()
        predictions = np.array([])
        true_labels = np.array([])
        with torch.no_grad(): # Deactivate gradients for the following code
            for data_inputs, data_labels in data_loader:

                # Determ£ine prediction of model on dev set
                data_inputs, data_labels = data_inputs.to(self.device), data_labels.to(self.device)
                preds = model(data_inputs)
                preds = torch.sigmoid(preds)
                predictions = np.concatenate((predictions,torch.argmax(preds, 1).data.cpu().numpy()),axis=0)
                true_labels = np.concatenate((true_labels,data_labels.data.cpu().numpy()),axis=0)
        model.train()
        return predictions,true_labels