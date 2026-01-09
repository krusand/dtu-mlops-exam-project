from exam_project.model import Model
from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
import torch
import transformers
import torchvision
import PIL

from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import os

# Function to open images
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')# grayscale==1 channel
# Transform: convert to tensor and optionally normalize

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(), # converts PIL image to tensor with values [0,1]
    ])
    return transform
def get_dataset(root, transform):
    dataset = DatasetFolder(
        root=root, # replace with your folder path
        loader=pil_loader,
        extensions=['jpg'],
        transform=transform
    )
    return dataset


def load_data():
    root = 'data/raw/datasets/msambare/fer2013/versions/1/'
    transform = get_transform()
    train_dataset = get_dataset(os.path.join(root,'train'), transform)
    test_dataset = get_dataset(os.path.join(root,'test'), transform)
    return train_dataset, test_dataset, test_dataset

    

def get_trainer(model, trainer_args):
    """
    Gets a Trainer object of either transformers or pytorch lightning

    params:
        model: The model class
        trainer_args: Training arguments
    """

    if model == "lightning":
        trainer = Trainer(**trainer_args)
    elif model == "huggingface":
        trainer = transformers.Trainer(**trainer_args)

    return trainer

def train():
    """
    Trains the model
    """
    trainer_args = dict()
    train, val, test = load_data()
    model = Model()
    trainer = get_trainer(model, trainer_args=trainer_args)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
    torch.save(model.state_dict(), "models/checkpoint.pth")



def load():
    model = Model()
    state_dict = torch.load("checkpoint.pth")
    model.load_state_dict(state_dict)


if __name__ == "__main__":
    img = torchvision.io.read_image(path="data/raw/datasets/msambare/fer2013/versions/1/test/angry/PrivateTest_88305.jpg")
    train()
