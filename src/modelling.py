# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data_modules import DataModule
from src.evaluation import Evaluation

# %%


def set_seed(seed: int = 42):
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
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class CCV1_Trainer:
    def __init__(
        self,
        data_model: DataModule,
        model: nn.Module,
        device: torch.device = device,

    ) -> None:
        """
        Jan
        Load the train/test/val data.
        :param DataModule data_model: instance where the 3 dataloader are available.
        :param nn.Module model: pytorch deep learning module
        :param torch.device device: used device for training
        """
        set_seed()
        self.data_model = data_model
        self.test_loader = data_model.test_dataloader()
        self.model = model
        self.device = device
        self.evaluation = Evaluation()

    def setup_wandb_run(self, project_name: str, run_group: str, fold: int, lr: float, num_epochs: int, model_architecture: str):
        """
        Thomas
        Sets a new run up (used for k-fold)
        :param str project_name: Name of the project in wandb.
        :param str run_group: Name of the project in wandb.
        :param str fold: number of the executing fold
        :param int lr: learning rate of the model
        :param int num_epochs: number of epochs to train
        :param str model_architecture: Modeltype (architectur) of the model
        """
        # init wandb
        self.run = wandb.init(
            settings=wandb.Settings(start_method="thread"),
            project=project_name,
            entity="deeptier",
            name=f"{fold}-Fold",
            group=run_group,
            config={
                "learning rate": lr,
                "epochs": num_epochs,
                "model architecture": model_architecture,
            }
        )

    def train_model(
        self,
        run_group: str,
        model_architecture: str,
        num_epochs: int,
        loss_module: nn = nn.CrossEntropyLoss(),
        test_model: bool = False,
        cross_validation: bool = True,
        project_name: str = "ccv1",
        batchsize_train_data: int = 64,
        num_workers: int = 16,
        lr: float = 1e-3
    ) -> None:
        """
        Jan
        To train a pytorch model.
        :param str run_group: Name of the run group (kfolds).
        :param str model_architecture: Modeltype (architectur) of the model
        :param int num_epochs: number of epochs to train
        :param nn.CrossEntropyLoss loss_module: Loss used for the competition
        :param int test_model: If true, it only loops over the first train batch and it sets only one fold. -> For the overfitting test.
        :param int cross_validation: If true, creates 5 cross validation folds to loop over, else only one fold is used for training
        :param str project_name: Name of the project in wandb.
        :param int batchsize: batchsize of the training data
        :param int num_workers: number of workers for the data loader (optimize if GPU usage not optimal) -> default 16
        :param int lr: learning rate of the model
        """

        # train loop over folds
        if cross_validation:
            n_folds = 5
        else:
            n_folds = 1

        for fold in tqdm(range(n_folds), unit="fold", desc="Fold-Iteration"):

            # setup a new wandb run for the fold -> fold runs are grouped by name
            self.setup_wandb_run(project_name, run_group,
                                 fold, lr, num_epochs, model_architecture)

            # prepare the kfold and dataloaders
            self.data_model.prepare_data(fold)
            self.train_loader = self.data_model.train_dataloader(
                batchsize_train_data, num_workers)
            self.val_loader = self.data_model.val_dataloader()

            # Overfitting Test for first batch
            if test_model:
                self.train_loader = [next(iter(self.train_loader))]

            # prepare the model
            model = self.model()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # training mode
            model.train()
            model.to(device)
            # train loop over epochs
            batch_iter = 1
            for epoch in tqdm(range(num_epochs), unit="epoch", desc="Epoch-Iteration"):
                loss_train = np.array([])
                label_train_data = np.empty((0, 8))
                pred_train_data = np.array([])

                # train loop over batches
                for batch in self.train_loader:

                    # calc gradient
                    data_inputs = batch["image"].to(device)
                    data_labels = batch["label"].to(device)

                    preds = model(data_inputs)
                    loss = loss_module(preds, data_labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    self.evaluation.per_batch(batch_iter, epoch, loss)

                    # data for evaluation
                    label_train_data = np.concatenate(
                        (label_train_data, data_labels.data.cpu().numpy()), axis=0
                    )
                    predict_train = torch.argmax(preds, 1).data.cpu().numpy()
                    pred_train_data = np.concatenate(
                        (pred_train_data, predict_train), axis=0
                    )
                    loss_train = np.append(loss_train, loss.item())

                    # iter next batch
                    batch_iter += 1

                # wandb per epoch
                pred_val, label_val = self.predict(model, self.val_loader)
                loss_val = loss_module(torch.tensor(
                    pred_val), torch.tensor(label_val))
                self.evaluation.per_epoch(
                    epoch,
                    loss_train.mean(),
                    pred_train_data,
                    label_train_data,
                    loss_val,
                    np.argmax(pred_val, axis=1),
                    label_val,
                )

            # wandb per model
            self.evaluation.per_model(
                label_val, pred_val, self.data_model.val.data)
            self.run.finish()
        # new model instance for a new k-fold
        self.model_fold5 = model

    def predict(self, model: nn.Module, data_loader: DataLoader):
        """
        Jan
        Prediction for a given model and dataset
        :param nn.Module model: pytorch deep learning module
        :param DataLoader data_loader: data for a prediction

        :return: predictions and true labels
        :rtype: np.array, np.array
        """
        model.eval()
        predictions = np.empty((0, 8))
        true_labels = np.empty((0, 8))
        with torch.no_grad():  # Deactivate gradients for the following code
            for batch in data_loader:

                # Determine prediction of model
                data_inputs = batch["image"].to(self.device)

                preds = model(data_inputs)
                preds = torch.sigmoid(preds)
                predictions = np.concatenate(
                    (predictions, preds.data.cpu().numpy()), axis=0
                )

                # checks if labels columns exists -> if not exists test batch
                if 'label' in batch.keys():
                    data_labels = batch["label"].to(self.device)
                    true_labels = np.concatenate(
                        (true_labels, data_labels.data.cpu().numpy()), axis=0
                    )
        model.train()
        return predictions, true_labels

    def submit_file(self, submit_name: str):
        """
        Jan
        Creates the file for the submission
        :param str submit_name: name of the file
        """
        # prediction off the test set
        prediction_test, _ = self.predict(self.model_fold5, self.test_loader)
        results_df = pd.DataFrame(
            prediction_test, columns=self.evaluation.classes)
        submit_df = pd.concat(
            [self.data_model.test.data.reset_index()["id"], results_df], axis=1
        )
        submit_df.set_index("id").to_csv(f"./data_submit/{submit_name}.csv")

# %%
