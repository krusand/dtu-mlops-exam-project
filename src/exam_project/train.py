from exam_project.model import BaseCNN
from exam_project.data import load_data

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import transformers
from typing import Annotated


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg):
    """
    Trains the model

    params: 
        cfg: config.yaml using Hydra
    """

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath='models/',
        filename='emotion-model-{epoch:02d}-{validation_loss:.2f}'
    )

    trainer_args = {"max_epochs": cfg.trainer.max_epochs
                    , 'accelerator': cfg.trainer.accelerator
                    , 'logger': WandbLogger(log_model=True, project=cfg.logger.wandb.project)
                    , 'log_every_n_steps': 5
                    , "callbacks": [checkpoint_callback]}
    
    train, val, test = load_data(processed_dir='data/processed/')
    train = torch.utils.data.DataLoader(train, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    
    model = instantiate(cfg.models)
    trainer = Trainer(**trainer_args)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)


if __name__ == "__main__":
    train()
