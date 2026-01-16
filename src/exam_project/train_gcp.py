from exam_project.model import BaseCNN
from exam_project.data import load_data

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import torch
import typer
from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated

# Check if a .env file exists in the project root (if statement so GCP ignores this)
dotenv_path = Path(__file__).parent.parent.parent / ".env"
print (dotenv_path)
if dotenv_path.exists():
    load_dotenv(dotenv_path)

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")

DATA_DIR = os.environ.get("DATA_DIR", "data/processed/")
MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "models")

ACCELERATOR = 'gpu' if torch.cuda.is_available() else 'cpu'
DEVICES=1

NUM_WORKERS = int(os.getenv("NUM_WORKERS","4"))
PERSISTENT_WORKERS = NUM_WORKERS>0

app = typer.Typer()

def get_trainer(trainer_args:dict)->pytorch_lightning.Trainer:
    """
    Returns PyTorch Lightning Trainer

    params:
        trainer_args: Training arguments
    """
    return Trainer(**trainer_args)

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
    #W&B
    if WANDB_PROJECT is None:
        raise RuntimeError("WANDB_PROJECT environment variable not set")
    wandb_logger = WandbLogger(log_model="all", 
                               project=WANDB_PROJECT,
                               entity=WANDB_ENTITY
                               )

    #Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        mode='min',#i.e. we are aiming for the minimum validation loss
        dirpath=MODEL_DIR,
        filename='emotion-model-{epoch:02d}-{validation_loss:.2f}',
        save_top_k=1
    )

    #Trainer args
    trainer_args = {'max_epochs': max_epochs,
                    'limit_train_batches': 0.05,
                    'accelerator':ACCELERATOR,
                    'devices': DEVICES,
                    'logger': wandb_logger,
                    'log_every_n_steps': 5,
                    "callbacks": [checkpoint_callback],
                    "enable_checkpointing": True}
    
    train, val, _ = load_data(processed_dir=DATA_DIR)
    #Note DataLoader spawns num_workers child processes
    train = torch.utils.data.DataLoader(train, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=batch_size)
    
    model = BaseCNN(lr=lr)
    trainer = get_trainer(trainer_args=trainer_args)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
    print(checkpoint_callback.best_model_path)
if __name__ == "__main__":
    app()
