import hydra
import os
import pytorch_lightning
import torch
import wandb

from exam_project.data import load_data
from google.cloud import storage
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

@rank_zero_only
def wandb_init(cfg, cfg_omega):
    return wandb.init(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.entity,
        job_type=cfg.logger.wandb.job_type,
        config=cfg_omega
    )

@rank_zero_only
def wandb_log_artifact(artifact, model_name):
    # Log the artifact
    wandb.log_artifact(artifact)
    logger.info("Artifact created and logged")
    logger.info("Linking artifact")

    # Link to model registry
    target_path = f"krusand-danmarks-tekniske-universitet-dtu-org/wandb-registry-fer-model/{model_name}"
    wandb.run.link_artifact(
        artifact=artifact,
        target_path=target_path,
        aliases=["latest", "staging"]
    )
    logger.info(target_path)
    logger.info("Artifact linked")


@hydra.main(config_path="configs", config_name="train", version_base=None)
def train(cfg):
    """
    Trains the model

    params: 
        cfg: .yaml using Hydra
    """

    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Add a log file to the logger
    logger.remove()
    logger.add(os.path.join(hydra_path, "loguru_logging.log"), level=cfg.debug.level)
    logger.info("Training script started")
    logger.debug(cfg)
    cfg_omega = OmegaConf.to_container(cfg)
    model_name = hydra.core.hydra_config.HydraConfig.get().runtime.choices.models

    # Initialise once across devices via @rank_zero_only
    run = wandb_init(cfg, cfg_omega)

    # Define directories for running either locally or on Vertex AI
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(cfg.data_paths.data_root,cfg.data_paths.processed_str))
    MODEL_DIR = os.environ.get("AIP_MODEL_DIR", cfg.model_paths.model_root)

    # Set random seed
    pytorch_lightning.seed_everything(cfg.hyperparameters.seed, workers=True)

    # Make model dir if it doesn't not already exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        mode='min',#i.e. we are aiming for the minimum validation loss
        dirpath=MODEL_DIR,
        filename='emotion-model-{epoch:02d}-{validation_loss:.2f}',
        save_top_k=1# The best model (lowest validation loss) is saved
    )

    #Note log_model="all" saves a model every epoch (x2 files per model: the model artifact and meta data).
    trainer_args = {"max_epochs": cfg.trainer.max_epochs
                    , 'accelerator': cfg.trainer.accelerator
                    , 'devices':cfg.trainer.devices
                    , 'strategy':"ddp"
                    , 'logger': WandbLogger(log_model=cfg.logger.wandb.log_model, project=cfg.logger.wandb.project)
                    , 'limit_train_batches': cfg.trainer.limit_train_batches
                    , 'limit_val_batches': cfg.trainer.limit_val_batches
                    , 'log_every_n_steps': cfg.trainer.log_every_n_steps
                    , "callbacks": [checkpoint_callback]}
    logger.debug(f"{trainer_args = }")
    logger.info("Finished cfg setup")

    logger.info("Loading model")
    model = instantiate(cfg.models)
    logger.info("Model loaded")
    trainer = Trainer(**trainer_args)
    NUM_WORKERS = max(1, os.cpu_count() // trainer.world_size)#Total_workers = devices*num_workers; no. of workers that feed batches of data
    PERSISTENT_WORKERS = NUM_WORKERS > 0 #Whether workers stay alive across epochs
    logger.info("Starting dataloading")
    train, val, test = load_data(processed_dir=DATA_DIR)
    train = torch.utils.data.DataLoader(train, shuffle=True, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=cfg.data.batch_size, pin_memory=cfg.trainer.accelerator in ("gpu","cuda"))
    val = torch.utils.data.DataLoader(val, shuffle=False, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=cfg.data.batch_size)
    test = torch.utils.data.DataLoader(test, shuffle=False, persistent_workers=PERSISTENT_WORKERS, num_workers=NUM_WORKERS, batch_size=cfg.data.batch_size)
    logger.info("Finished dataloading")

    logger.info("Model fitting started")
    trainer.fit(model=model, train_dataloaders=train, val_dataloaders=val)
    logger.info("Model fitting finished")
    # Save and log the best model to model registry
    best_model_path = checkpoint_callback.best_model_path
    logger.info(f"{best_model_path = }")

    logger.info("Creating artifact")
    # Create an artifact
    artifact = wandb.Artifact(
        name=f"emotion-model-{model_name}",
        type="Model",
        description="Emotion recognition model",
        metadata={'architecture': model_name, 'device': cfg.trainer.accelerator}
    )
    logger.info(artifact)
    # Add the model file to the artifact
    if best_model_path.startswith("gs://"): #W&B cannot add unless file is local
        local_model_path = "/tmp/" + os.path.basename(best_model_path)  # Temp local path
        logger.info(f"{local_model_path = }")
        
        # Download from GCS
        client = storage.Client()
        bucket_name, blob_path = best_model_path[5:].split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_model_path)
        artifact.add_file(local_model_path)  # update to local path
    else:
        artifact.add_file(best_model_path)
    
    # Log model artifact
    wandb_log_artifact(artifact, model_name)
    run.finish()
    logger.info("Training script finished")

if __name__ == "__main__":
    train()
