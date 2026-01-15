from exam_project.model import BaseCNN, BaseANN
from exam_project.data import load_data
from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
import torch
import transformers
import torchvision
import PIL

import typer
from typing import Annotated

from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

app = typer.Typer()


def get_trainer(model, trainer_args):
    """
    Gets a Trainer object of either transformers or pytorch lightning

    params:
        model: The model class
        trainer_args: Training arguments
    """
    return Trainer(**trainer_args)

    # IMPLEMENT WHEN WE GET HUGGINGFACE MODELS
    if model == "lightning":
        trainer = Trainer(**trainer_args)
    elif model == "huggingface":
        trainer = transformers.Trainer(**trainer_args)

    return trainer

@app.command()
def train(
        max_epochs: Annotated[int, typer.Option("--max-epochs", "-max_e")] = 1,
        lr: Annotated[float, typer.Option("--learning-rate", "-lr")] = 1e-3,
        batch_size: Annotated[int, typer.Option("--batch-size", "-bs")] = 128
    ):
    """
    Trains the model

    params:
        max_epochs (int): The number of epochs the models runs for
        lr (float): Learning rate of gradient descent method
        batch_size: The number of images in a batch
    """
    trainer_args = {"max_epochs": max_epochs,'limit_train_batches': 0.05, 'accelerator': DEVICE}
    train, val, test = load_data(processed_dir='data/processed/')
    train = torch.utils.data.DataLoader(train, persistent_workers=True, num_workers=9, batch_size=batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=True, num_workers=9, batch_size=batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=True, num_workers=9, batch_size=batch_size)
    
    model = BaseANN(lr=lr)
    trainer = get_trainer(model, trainer_args=trainer_args)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
    torch.save(model.state_dict(), "models/checkpoint.pth")



def load():
    model = BaseANN()
    state_dict = torch.load("checkpoint.pth")
    model.load_state_dict(state_dict)


if __name__ == "__main__":
    app()
