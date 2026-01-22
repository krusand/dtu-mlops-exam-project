from exam_project.data import load_data
from exam_project.model import BaseANN, BaseCNN, ViTClassifier
from exam_project.utils import validate_environment, load_model_from_wandb, get_device_from_artifact

import os
import time
import torch
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

def test_model_speed() -> bool:
    """
    Tests to see if model can predict 100 values pr. second
    
    Returns:
        should_promote (bool): True if can predict >100 pr. second, else False
    """
    logger.info("Starting testing model speed")
    staging_model, staging_artifact = load_model_from_wandb(artifact=os.getenv("MODEL_NAME"), alias='staging')
    start = time.time()
    logger.info(f"{start = }")
    for _ in range(100):
        staging_model(torch.rand(1, 1, 48, 48).to(get_device_from_artifact(staging_artifact)))
    end = time.time()
    should_promote = end - start < 1
    logger.info(f"{end = }")
    logger.info(f"Time diff: {(end-start)}")
    logger.info(f"{should_promote = }")
    logger.info("Finished testing model speed")
    return should_promote

def evaluate_model(model: BaseANN | BaseCNN | ViTClassifier, test_dataloader: torch.utils.data.DataLoader, device: str) -> float:
    """
    Evaluates model

    Params:
        - model (BaseANN | BaseCNN | ViTClassifier): The loaded, trained model
        - test_dataloader (torch.utils.data.DataLoader): A DataLoader object with test data
        - device (str): The device

    Returns:
        - test_acc (float): The test accuracy of the model
    """
    logger.info("Evaluating model")

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

    logger.info(f"Number of predictions: {len(y_pred.tolist())}")
    
    test_acc = accuracy_score(y_true.tolist(), y_pred.tolist())
    
    logger.info(f"{test_acc = }")
    logger.info("Finished evaluating model")
    
    return test_acc


def test_staging_against_production_model() -> bool:
    """
    Tests accuracy of staging model against the current production model
    
    Returns:
        - should_promote (bool): True if staging accuracy is better than production accuracy
    """
    logger.info("Testing staging model against production model")
    staging_model, staging_artifact = load_model_from_wandb(os.getenv("MODEL_NAME"), alias='staging')
    production_model, production_artifact = load_model_from_wandb(os.getenv("MODEL_NAME"), alias='production')
    _, _, test = load_data(processed_dir="data/processed/")
    test = torch.utils.data.DataLoader(test, batch_size=64)

    staging_accuracy = evaluate_model(staging_model, test, get_device_from_artifact(staging_artifact))
    production_accuracy = evaluate_model(production_model, test, get_device_from_artifact(production_artifact))
    should_promote = staging_accuracy > production_accuracy

    logger.info(f"{staging_accuracy = }")
    logger.info(f"{production_accuracy = }")
    logger.info(f"{should_promote = }")
    logger.info("Finished testing staging model against production model")
    return should_promote
    
def main():
    logger.info("Stage and checking model")

    validate_environment()
    should_promote = all([test_staging_against_production_model()])
    should_promote = 'true' if should_promote else 'false' # Used for better yaml handling
        
    if "GITHUB_OUTPUT" in os.environ:
        logger.info("GITHUB_OUTPUT in environment, writing to it")
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"promote={should_promote}", file=f)
    else:
        # Fallback for local testing
        logger.info(f"Not in CI/CD. Output would be: promote={should_promote}")
    
    logger.info("Stage and checking model finished")


if __name__ == '__main__':
    main()



