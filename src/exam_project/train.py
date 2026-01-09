from exam_project.model import Model
from exam_project.data import MyDataset
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
    train, val, test = MyDataset("data/raw")
    model = Model()
    trainer = get_trainer()
    trainer.fit(model=model)
    # add rest of your training code here


if __name__ == "__main__":

    train()
