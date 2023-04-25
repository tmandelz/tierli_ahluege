# %%
from torchvision import transforms
from torch import nn
# %%


class None_Transform(nn.Module):
    """
    Is used as a None Transform, only forwards the inputs
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


# %%
class CCV1Transformer:
    """ """

    def __init__(
        self,
        data_augmentation_transformer: transforms.Compose,
        preprocessing_transformer: str,
        pretrained_transformer: str,
    ) -> None:
        """
        :param torchvision.transforms data_augmentation_transformer: transfomation steps for data_augmentation
        :param str preprocessing_transformer: string which defines a preset for all our transformation steps (resizing etc.)
        :param str pretrained_transformer: string which defines a preset for the pretrained transfer model settings
        """
        self.pretrained_transformer = pretrained_transformer
        self.preprocessing_transformer = preprocessing_transformer
        self.data_augmentation_transformer = data_augmentation_transformer

        # determine preprocessing steps
        if preprocessing_transformer == "standard":
            self.preprocessing_transformer = transforms.Compose(
                [
                    transforms.Resize((224, 224), antialias=True),
                ]
            )
        elif preprocessing_transformer == "overfitting":
            self.preprocessing_transformer = transforms.Compose(
                [
                    transforms.Resize((20, 20), antialias=True),
                ]
            )
        elif preprocessing_transformer == "model_specific":
            self.preprocessing_transformer = transforms.Compose(
                [
                    None_Transform()
                ]
            )

        # determine transformation steps for pretrained Models
        if pretrained_transformer == "efficientnet":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (256, 256), antialias=True, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif pretrained_transformer == "resnet":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (232, 232), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif pretrained_transformer == "mlp":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]
            )
        elif pretrained_transformer == "convnext":
            self.pretrained_transformer = transforms.Compose(
                [
                    transforms.Resize(
                        (236, 236), antialias=True, interpolation=transforms.InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])

    def getCompose(self, turn_off_to_tensor: bool = False):
        """
        :param turn_off_to_tensor:bool: don't use transforms.ToTensor() for certain augmentations
        """
        if turn_off_to_tensor:
            first_trans = None_Transform()
            sec_trans = transforms.ToTensor()
        else:
            first_trans = transforms.ToTensor()
            sec_trans = None_Transform()
        return transforms.Compose(
            [
                # add a to Tensor in front off all augmentations
                first_trans,
                # first execute the augmenation steps
                self.data_augmentation_transformer,
                # for special data augmentation methods
                sec_trans,
                # second execute the preprocessing steps
                self.preprocessing_transformer,
                # third execute the pretrained model step
                self.pretrained_transformer,
                # transforms.ToTensor()
            ]
        )
