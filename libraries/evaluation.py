# %%
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from PIL import Image


# %%
classes = [
    "antelope_duiker",
    "bird",
    "blank",
    "civet_genet",
    "hog",
    "leopard",
    "monkey_prosimian",
    "rodent",
]


class Evaluation:
    def __init__(self, data_classes: list = classes) -> None:
        """
        Jan
        :param list data_classes: list of true labels (convert from int to str)
        """
        self.classes = data_classes

    def per_batch(self, loss_batch) -> None:
        """
        Thomas
        Logs the loss of a batch
        """
        wandb.log({"loss batch": loss_batch})

    def per_epoch(
        self,
        loss_train: float,
        pred_train: np.array,
        label_train: np.array,
        loss_val: float,
        pred_val: np.array,
        label_val: np.array,
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
        f1_train = {
            f"train f1_score von {self.classes[animal]}": f1_score(
                label_train[:, animal], pred_train == animal
            )
            for animal in range(len(self.classes))
        }
        f1_test = {
            f"validation f1_score von {self.classes[animal]}": f1_score(
                label_val[:, animal], pred_val == animal
            )
            for animal in range(len(self.classes))
        }
        log = {"Loss train": loss_train, "Loss val": loss_val}
        wandb.log({**f1_train, **f1_test, **log})

    def per_model(self, label_val, pred_val, val_data) -> None:
        """
        Jan
        wandb log of a confusion matrix and plots of wrong classified animals
        :param np.array label_val: labels of the validation
        :param np.array pred_val: prediction of the validation
        :param pd.dataframe val_data: validation data
        """
        self.true_label = np.argmax(label_val, axis=1)
        self.true_pred = np.argmax(pred_val, axis=1)
        wrong_classified = np.where(self.true_label != self.true_pred)[0]

        self.plot_16_animals(
            np.random.choice(wrong_classified, replace=False, size=16), val_data
        )

        wandb.log(
            {
                "confusion matrix": wandb.sklearn.plot_confusion_matrix(
                    self.true_label, self.true_pred, self.classes
                ),
                "wrong prediction": plt,
            }
        )
        plt.close()

        data_wrong_class = val_data.iloc[wrong_classified]
        site_most_wrong = data_wrong_class[
            data_wrong_class["site"] == data_wrong_class["site"].value_counts().index[0]
        ]
        if len(site_most_wrong) < 16:
            self.plot_16_animals(range(len(site_most_wrong)), data_wrong_class)
        else:
            self.plot_16_animals(
                np.random.choice(range(len(site_most_wrong)), size=16, replace=False),
                data_wrong_class,
            )

        plt.suptitle("worst site: " + str(data_wrong_class["site"][0]), size=120)
        wandb.log({"Bad site": plt})
        plt.close()

    def plot_16_animals(self, index: np.array, data: pd.DataFrame):
        """
        Jan
        plot 16 animals
        :param np.array index: index of the choosen animals
        :param pd.DataFrame data: data with the filepath of the images
        """
        fig = plt.figure(figsize=(120, 90), dpi=20)
        for n, variable in enumerate(index):
            ax = fig.add_subplot(4, 4, n + 1)
            datapoint = data.iloc[variable]
            ax.imshow(Image.open("./competition_data/" + datapoint["filepath"]))
            ax.set_title(
                f"{self.classes[self.true_pred[variable]]} anstatt {self.classes[self.true_label[variable]]}",
                size=60,
            )


# %%
