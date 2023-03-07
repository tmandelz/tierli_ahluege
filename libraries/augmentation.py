from torchvision import transforms
# %%

class data_augmentation():
    def __init__(self) -> None:
        transformer = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
