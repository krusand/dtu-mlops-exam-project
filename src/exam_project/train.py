from exam_project.model import BaseCNN
from exam_project.data import load_data

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import transformers
import typer
from typing import Annotated
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    wandb_logger = WandbLogger(log_model="all", project=project)
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath='models/',
        filename='emotion-model-{epoch:02d}-{validation_loss:.2f}'
    )

    trainer_args = {"max_epochs": max_epochs
                    ,'limit_train_batches': 0.05
                    , 'accelerator': DEVICE
                    , 'logger': wandb_logger
                    , 'log_every_n_steps': 5
                    , "callbacks": [checkpoint_callback]}
    
    train, val, test = load_data(processed_dir='data/processed/')
    train = torch.utils.data.DataLoader(train, persistent_workers=True, num_workers=9, batch_size=batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=True, num_workers=9, batch_size=batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=True, num_workers=9, batch_size=batch_size)
    
    model = BaseCNN(lr=lr)
    trainer = get_trainer(model, trainer_args=trainer_args)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
    print(checkpoint_callback.best_model_path)
if __name__ == "__main__":
    app()
