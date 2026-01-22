from exam_project.model import BaseCNN, BaseANN, ViTClassifier

# Imports the Google Cloud client library
from google.cloud import storage
from dotenv import load_dotenv
import os
import wandb
import pytorch_lightning as pl
import torch

load_dotenv()

MODELS = {
    'ann': BaseANN,
    'cnn': BaseCNN,
    'vit': ViTClassifier
}

def delete_blob(bucket, blob_name):
    blob = bucket.blob(blob_name)
    generation_match_precondition = True

    blob.reload()
    generation_match_precondition = blob.generation

    blob.delete(if_generation_match=generation_match_precondition)

def write_blob(bucket, blob_name):

    blob = bucket.blob(blob_name)
    blob.upload_from_filename("production_model.ckpt")

def save_model_to_checkpoint(model):
    checkpoint = {
        "state_dict": model.state_dict(),
        "hyper_parameters": getattr(model, 'hparams', {}),
        "pytorch-lightning_version": pl.__version__,
    }
    torch.save(checkpoint, "production_model.ckpt")

def load_model_from_wandb(artifact: str, alias: str = 'production'):

    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY_ORG")
                   , "project": os.getenv("WANDB_PROJECT")},
    )
    
    artifact_name_version = f"{os.getenv("MODEL_NAME")}"
    artifact_name, artifact_version = artifact_name_version.split(":")
    artifact = api.artifact(f"{artifact_name}:{alias}", type="Model")
    artifact.download(root="./artifacts")
    file_name = artifact.files()[0].name
    print(file_name)
    model = MODELS[os.getenv("MODEL_ARCHITECTURE")]
    return model.load_from_checkpoint(f"./artifacts/{file_name}"), artifact

def main():    
    storage_client = storage.Client(project="decent-seeker-484209-j2")
    bucket = storage_client.bucket("dtu-mlops-exam-project-data")

    model_architecture = os.getenv("MODEL_ARCHITECTURE")

    # Find model to delete
    blobs = list(bucket.list_blobs(prefix=f"models/{model_architecture}"))
    ckpts = [blob for blob in blobs if blob.name.endswith(".ckpt")]
    #assert len(ckpts) == 1, "More than one ckpt"
    ckpt = ckpts[0]

    model, artifact = load_model_from_wandb(os.getenv("MODEL_NAME"))
    # Upload new production model
    save_model_to_checkpoint(model)
    write_blob(bucket, f"models/{model_architecture}/{model_architecture}_production_model.ckpt")

    # Delete model
    delete_blob(bucket, ckpt.name)

if __name__ == '__main__':
    main()
