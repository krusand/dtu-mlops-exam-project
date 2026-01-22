from pathlib import Path
from typing import List, Tuple, Union

from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import typer

from exam_project.data import load_data
from exam_project.model import BaseANN


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


def get_predictions(
        model: Union[Module, LightningModule], 
        data_loader: DataLoader, 
        device: torch.device,
        n: int,
        ) -> Tuple[List[int], List[int]]:
    """
    Evaluate a PyTorch or Lightning model on a dataset and return predictions and true labels.

    Args:
        model:          nn.Module or LightningModule
        data_loader:    DataLoader providing (inputs, targets)
        device:         torch.device to run the model on
        n:              Number of test batches to process.

    Returns:
        Tuple containing:
            - y_pred:   list of predicted class indices
            - y_true:   list of true class indices
    """
    y_pred = []
    y_true = []

    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i >= n:
                break
            else:
                data, target = data.to(device), target.to(device)
                y_true.append(target.item())
                output = model(data)
                predicted = output.argmax(dim=1)
                y_pred.append(predicted.item())
    
    return y_pred, y_true


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
    n = len(test_loader)
    y_pred, y_true = get_predictions(model, test_loader, device, n)

    # computing evaluation metrics
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
    