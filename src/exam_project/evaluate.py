from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader

from exam_project.data import load_data
from exam_project.model import BaseANN, BaseCNN

import typer
from typing import Annotated

ROOT = Path(__file__).resolve().parents[2]    # go two levels up to project root
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

app = typer.Typer()

def load_model(model_file_name: str = "checkpoint.pth", device: str = DEVICE) -> None:
    """
    Loads a trained image classification model.

    Params: 
    - model_file_name:      Name of file containing trained model object.
    - device:               Device on which to store the test data.

    Returns:
    - model:                Retrieved model object.
    """
    model = BaseANN()   
    model_path = str(ROOT / "models" / f"{model_file_name}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model

@app.command()
def evaluate_model(model_file_name: str = "checkpoint.pth", 
                   test_data_path: str = "data/processed/", 
                   device: str = DEVICE
                   ) -> dict:
    """
    Evaluates a trained image classification model.

    Params:
    - model_file_name:  Path to model.
    - test_data_path:   Path to test data.
    - device:           Device on which to store the test data.

    Returns:
    - eval_dict:        Dictionary containing evaluation metrics.
    """
    # loading trained model
    model = load_model(model_file_name, device)
    
    # loading test data
    _, _, test = load_data(processed_dir=test_data_path)

    # dataloader
    test_loader = DataLoader(test, persistent_workers=True, num_workers=9)

    # making predictions on the test set one image at a time
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_true.append(target.item())
            output = model(data)
            predicted = output.argmax(dim=1)
            y_pred.append(predicted.item())

    test_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y_true, y_pred)
    eval_dict = {"Test accuracy": test_acc,
                 "Macro F1": macro_f1,
                 "Weighted F1": weighted_f1,
                 "Confusion matrix": conf_matrix,
                 }

    return eval_dict


def print_eval_dict():
    eval_dict = evaluate_model(model_file_name="checkpoint.pth",
                               test_data_path="data/processed/",   # TODO: add model and test_data_path to config file
                               device=DEVICE)
    print(eval_dict)

if __name__ == "__main__":
    app()
    