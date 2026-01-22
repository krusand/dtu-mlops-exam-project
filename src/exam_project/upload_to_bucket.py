from exam_project.model import BaseCNN, BaseANN, ViTClassifier
from exam_project.utils import validate_environment, load_model_from_wandb, get_device_from_artifact

# Imports the Google Cloud client library
from google.cloud import storage
from dotenv import load_dotenv
import os
import pytorch_lightning as pl
import torch
from loguru import logger

load_dotenv()

def write_blob(bucket: storage.Client.bucket, blob_name: str, path_to_model: str) -> None:
    """
    Writes file to blob
    
    Params:
        - bucket (storage.Client.bucket): The storage.Client.bucket initialised to our bucket.
        - blob_name (str): The name of the blob (folder/file) to write to
        - path_to_model (str): The path to the {model}.ckpt
    
    Returns:
        - None
    """

    logger.info("Writing to bucket")
    logger.info(f"Writing to blob: {blob_name}")

    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_model)
    
    logger.info("ckpt uploaded")

def save_model_to_checkpoint(model: BaseANN | BaseCNN | ViTClassifier, path_to_model: str) -> None:
    """
    Saves a loaded model to checkpoint

    Params:
        - model (BaseANN | BaseCNN | ViTClassifier): The model loaded from artifact checkpoint
        - path_to_model (str): Path to save model
    
    Returns:
        - None
    """

    logger.info("Saving model to checkpoint")

    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": getattr(model, 'hparams', {}),
        "pytorch-lightning_version": pl.__version__,
    }

    torch.save(checkpoint, path_to_model)
    
    logger.info(f"Model saved to {path_to_model}")

def test_model(model, artifact) -> None:
    """Tests model on random noise to see if it can predict"""

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

    model, artifact = load_model_from_wandb(artifact=os.getenv("MODEL_NAME"), alias='production')

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
