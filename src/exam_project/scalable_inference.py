from pathlib import Path
from typing import List, Tuple, Union

from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time
import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import typer

from exam_project.data import load_data
from exam_project.model import ViTClassifier

#import torch.quantization as tq
from loguru import logger

def apply_global_pruning(
    model: torch.nn.Module,
    amount: float = 0.2,
    pruning_method=prune.L1Unstructured
) -> torch.nn.Module:
    """
    Apply global unstructured pruning to Linear and Conv2d layers.

    Params:
    - model: trained model
    - amount: fraction of weights to prune globally (0.0–1.0)
    - pruning_method: pruning class (default L1)

    Returns:
    - pruned model
    """
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, "weight"))
        elif isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, "weight"))

    # Create pruning mask
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=amount,
    )

    # Apply pruning mask
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    return model



ROOT = Path(__file__).resolve().parents[2]    # go two levels up to project root
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
#DEVICE = 'cpu' #mps quantization doesn't work, so stick with cpu
# Set quantization engine
#torch.backends.quantized.engine = "qnnpack"

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
    #model = BaseANN()
    model = ViTClassifier()
    model_path = str(ROOT / "models" / f"{model_file_name}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model


def get_predictions(
        model: Union[Module, LightningModule], 
        data_loader: DataLoader, 
        device: torch.device
        ) -> Tuple[List[int], List[int]]:
    """
    Evaluate a PyTorch or Lightning model on a dataset and return predictions and true labels.

    Args:
        model: nn.Module or LightningModule
        data_loader: DataLoader providing (inputs, targets)
        device: torch.device to run the model on

    Returns:
        Tuple containing:
            - y_pred: list of predicted class indices
            - y_true: list of true class indices
    """
    y_pred = []
    y_true = []

    model.eval()

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            y_true.extend(target.tolist())
            output = model(data)
            predicted = output.argmax(dim=1)
            y_pred.extend(predicted.tolist())

    
    return y_pred, y_true


@app.command()
def evaluate_model(model_file_name: str = "checkpoint.pth", 
                   test_data_path: str = "data/processed/",
                   pruning_amount: float = 0.2,
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
    logger.info("Loading model...")
    model = load_model(model_file_name, device)

    #Compile model
    logger.info("Compiling model...")
    model = torch.compile(model,   
                          #mode="reduce-overhead",fullgraph=False)
                          backend="aot_eager")

    logger.info("Warm up model...")
    dummy = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = model(dummy)



    #model = tq.quantize_dynamic(
    #                model,
    #                {torch.nn.Linear},
    #                dtype=torch.qint8
    #            )

    # apply pruning (lower accuracy and not much performance gain; don't use)
    #model = apply_global_pruning(model, amount=pruning_amount)
    
    # loading test data
    logger.info("Loading data...")
    _, _, test = load_data(processed_dir=test_data_path)

    # Load data in batches to speed up inference
    test_loader = DataLoader(test,batch_size=16,
                                num_workers=4,
                                pin_memory=True)#, persistent_workers=False, num_workers=0)

    # making predictions on the test set one image at a time
    logger.info("Doing inference...")
    start = time.time()
    y_pred, y_true = get_predictions(model, test_loader, device)
    logger.info(f"Inference time: {time.time()-start}")
    print (y_pred)
    print (y_true)

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
    print (eval_dict)

    return eval_dict


def print_eval_dict():
    eval_dict = evaluate_model(model_file_name="checkpoint.pth",
                               test_data_path="data/processed/",   # TODO: add model and test_data_path to config file
                               device=DEVICE)
    print(eval_dict)

if __name__ == "__main__":
    app()
    