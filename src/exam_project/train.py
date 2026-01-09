from exam_project.model import Model
from exam_project.data import load_data
from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
import torch
import transformers

def get_trainer(model, trainer_args):
    """
    Gets a Trainer object of either transformers or pytorch lightning

    params:
        model: The model class
        trainer_args: Training arguments
    """

    if model is "lightning":
        trainer = Trainer(**trainer_args)
    elif model is "huggingface":
        trainer = transformers.Trainer(**trainer_args)

    return trainer

def train():
    """
    Trains the model
    """
    trainer_args = dict()
    train, val, test = load_data("data/raw")
    model = Model()
    trainer = get_trainer(model, trainer_args=trainer_args)
    trainer.fit(model=model)
    torch.save(model.state_dict(), "models/checkpoint.pth")


def load():
    model = Model()
    state_dict = torch.load("checkpoint.pth")
    model.load_state_dict(state_dict)


if __name__ == "__main__":

    train()
