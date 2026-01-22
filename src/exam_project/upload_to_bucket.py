from exam_project.model import BaseCNN, BaseANN, ViTClassifier

# Imports the Google Cloud client library
from google.cloud import storage
from dotenv import load_dotenv
import os
import wandb
import pytorch_lightning as pl
import torch
from loguru import logger
import sys

load_dotenv()

MODELS = {
    'ann': BaseANN,
    'cnn': BaseCNN,
    'vit': ViTClassifier
}


def validate_environment() -> None:
    def validate_environment_var(environment_var: str) -> bool:
        env_var = os.getenv(environment_var)
        return all([env_var is not None, env_var != ""])
    
    logger.info("Asserting environment variables")

    assert validate_environment_var("MODEL_ARCHITECTURE"), 'Environment variable "MODEL_ARCHITECTURE" is None or ""'
    assert validate_environment_var("MODEL_NAME"), 'Environment variable "MODEL_NAME" is None or ""'
    assert validate_environment_var("WANDB_API_KEY"), 'Environment variable "WANDB_API_KEY" is None or ""'
    assert validate_environment_var("WANDB_ENTITY_ORG"), 'Environment variable "WANDB_ENTITY_ORG" is None or ""'
    assert validate_environment_var("WANDB_PROJECT"), 'Environment variable "WANDB_PROJECT" is None or ""'

    logger.info("All environment variables exist")

def write_blob(bucket: storage.Client.bucket, blob_name: str, path_to_model: str) -> None:
    logger.info("Writing to bucket")
    logger.info(f"Writing to blob: {blob_name}")

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_model)
    
    logger.info("ckpt uploaded")

def save_model_to_checkpoint(model: BaseANN | BaseCNN | ViTClassifier, path_to_model: str) -> None:
    logger.info("Saving model to checkpoint")

    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": getattr(model, 'hparams', {}),
        "pytorch-lightning_version": pl.__version__,
    }

    torch.save(checkpoint, path_to_model)
    
    logger.info(f"Model saved to {path_to_model}")

def load_model_from_wandb(artifact: str, alias: str = 'production'):
    logger.info("Loading model artifact from WandB")
    logger.info("Function inputs:")
    logger.info(f"{artifact = }, {alias = }")
    
    model_architecture = os.getenv("MODEL_ARCHITECTURE")
    model = MODELS.get(model_architecture)

    logger.info(f"{model = }")    
    assert model_architecture in list(MODELS.keys()), f"Model architecture '{model_architecture}' not supported."

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY_ORG")
                   , "project": os.getenv("WANDB_PROJECT")},
    )

    logger.info("API connection established")
    
    artifact_name_version = f"{os.getenv("MODEL_NAME")}"
    logger.info(f"{artifact_name_version = }")

    artifact_name, artifact_version = artifact_name_version.split(":")
    artifact = api.artifact(f"{artifact_name}:{alias}", type="Model")

    logger.info("Downloading artifact")
    artifact.download(root="./artifacts")
    logger.info("Artifact downloaded")

    file_name = artifact.files()[0].name

    logger.info(f"{file_name = }")

    return model.load_from_checkpoint(f"./artifacts/{file_name}"), artifact


def test_model(model, artifact):
    def get_device_from_artifact(artifact):
        return artifact.metadata.get("device")

    model(torch.rand(1, 1, 48, 48).to(get_device_from_artifact(artifact)))


def main():    
    logger.info("Starting upload of production model")

    validate_environment()
    storage_client = storage.Client(project="decent-seeker-484209-j2")
    bucket = storage_client.bucket("dtu-mlops-exam-project-data")
    
    logger.info("Connected to bucket")

    model_architecture = os.getenv("MODEL_ARCHITECTURE")

    logger.info(f"{model_architecture = }")

    blobs = list(bucket.list_blobs(prefix=f"models/{model_architecture}"))

    # There should be a maximum of one {model}.ckpt in each model folder
    assert len(blobs)-1 <= 1, "There should be only one ckpt in folder"
    if len(blobs)-1 == 0:
        logger.warning("No models found in folder, uploading new model")
    else:
        logger.info("Model already exist in folder. Overwriting model in folder")

    model, artifact = load_model_from_wandb(os.getenv("MODEL_NAME"))

    # Overwrite production model
    save_model_to_checkpoint(model=model, path_to_model="production_model.ckpt")
    write_blob(bucket=bucket
            , blob_name=f"models/{model_architecture}/{model_architecture}_production_model.ckpt"
            , path_to_model="production_model.ckpt")
    
    blobs = list(bucket.list_blobs(prefix=f"models/{model_architecture}"))

    assert len(blobs)-1 == 1, f"Model folder contains {len(blobs)-1} elements, it should contain 1"
    
    logger.info("Finished upload of production model")

if __name__ == '__main__':
    main()
