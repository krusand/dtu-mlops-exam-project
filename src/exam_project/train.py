from exam_project.data import load_data

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import wandb
from omegaconf import OmegaConf

@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg):
    """
    Trains the model

    params: 
        cfg: .yaml using Hydra
    """

    cfg_omega = OmegaConf.to_container(cfg)


    run = wandb.init(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.entity,
        job_type=cfg.logger.wandb.job_type,
        config=cfg_omega
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        dirpath='models/',
        filename='emotion-model-{epoch:02d}-{validation_loss:.2f}'
    )

    trainer_args = {"max_epochs": cfg.trainer.max_epochs
                    , 'accelerator': cfg.trainer.accelerator
                    , 'logger': WandbLogger(log_model=cfg.logger.wandb.log_model, project=cfg.logger.wandb.project)
                    , 'log_every_n_steps': cfg.trainer.log_every_n_steps
                    , "callbacks": [checkpoint_callback]}
    
    train, val, test = load_data(processed_dir='data/processed/')
    train = torch.utils.data.DataLoader(train, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    val = torch.utils.data.DataLoader(val, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    test = torch.utils.data.DataLoader(test, persistent_workers=True, num_workers=9, batch_size=cfg.data.batch_size)
    
    model = instantiate(cfg.models)
    trainer = Trainer(**trainer_args)
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)

    # Save and log the best model to model registry
    best_model_path = checkpoint_callback.best_model_path
    
    # Create an artifact
    artifact = wandb.Artifact(
        name=f"emotion-model-{cfg.models._target_}",
        type="model",
        description="Emotion recognition model"
    )
    
    # Add the model file to the artifact
    artifact.add_file(best_model_path)
    
    # Log the artifact
    wandb.log_artifact(artifact)
    
    # Link to model registry
    wandb.run.link_artifact(
        artifact=artifact,
        target_path="krusand-danmarks-tekniske-universitet-dtu-org/wandb-registry-fer-model/Model new",
        aliases=["latest"]
    )
    run.finish()

if __name__ == "__main__":
    train()
