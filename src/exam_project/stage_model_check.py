from exam_project.data import load_data
from exam_project.model import BaseANN, BaseCNN, ViTClassifier
from exam_project.utils import validate_environment

import wandb
import os
import time
import torch
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
load_dotenv()
MODELS = {
    'ann': BaseANN,
    'cnn': BaseCNN,
    'vit': ViTClassifier
}

def load_model(artifact: str, alias: str = 'staging'):

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
    model = MODELS[os.getenv("MODEL_ARCHITECTURE")]
    return model.load_from_checkpoint(f"./artifacts/{file_name}"), artifact

def get_device_from_artifact(artifact):
    return artifact.metadata.get("device")

def test_model_speed():
    staging_model, staging_artifact = load_model(os.getenv("MODEL_NAME"), alias='staging')
    start = time.time()
    for _ in range(100):
        staging_model(torch.rand(1, 1, 48, 48).to(get_device_from_artifact(staging_artifact)))
    end = time.time()
    should_promote = end - start < 1
    return should_promote

def evaluate_model(model, test_dataloader, device):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = output.argmax(dim=1)
            y_true.append(target)
            y_pred.append(predicted)
    
    y_true = torch.cat(y_true).to(device)
    y_pred = torch.cat(y_pred).to(device)

    test_acc = accuracy_score(y_true.tolist(), y_pred.tolist())
    return test_acc


def test_staging_against_production_model():
    staging_model, staging_artifact = load_model(os.getenv("MODEL_NAME"), alias='staging')
    production_model, production_artifact = load_model(os.getenv("MODEL_NAME"), alias='production')
    _, _, test = load_data(processed_dir="data/processed/")
    test = torch.utils.data.DataLoader(test, batch_size=64)


    staging_accuracy = evaluate_model(staging_model, test, get_device_from_artifact(staging_artifact))
    production_accuracy = evaluate_model(production_model, test, get_device_from_artifact(production_artifact))
    print(f"{staging_accuracy = }")
    print(f"{production_accuracy = }")
    should_promote = staging_accuracy > production_accuracy
    return should_promote
    
def main():
    validate_environment()
    should_promote = all([test_staging_against_production_model()])
    should_promote = 'true' if should_promote else 'false' # Used for better yaml handling
        
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"promote={should_promote}", file=f)
    else:
        # Fallback for local testing
        print(f"Not in CI/CD. Output would be: promote={should_promote}")


if __name__ == '__main__':
    main()



