'''
from pathlib import Path

import typer
from torch.utils.data import Dataset
import os
import sys 

import kagglehub
import pandas as pd
import shutil
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import *


def download_dataset() -> str:
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")

    print("Path to dataset files:", path)
    return path

def move_dataset_from_cache_to_folder(path_to_cache: str, path_to_folder: str) -> None:
    shutil.copytree(path_to_cache, path_to_folder, dirs_exist_ok=True)
    shutil.rmtree(path_to_folder / "data", ignore_errors=True)

def convert_to_parquet() -> None:

    for file in tqdm(os.listdir(DATA_RAW_DIR)):
        file_name, file_extension = file.split(".")
        file_extension = "."+(file_extension)
        pd.read_csv(DATA_RAW_DIR / (file_name + file_extension)).to_parquet(DATA_CLEANED_DIR / (file_name + ".pq"))

if __name__ == "__main__":
    path_to_cache = download_dataset()
    move_dataset_from_cache_to_folder(path_to_cache=path_to_cache, path_to_folder=DATA_RAW_DIR)
    convert_to_parquet()



class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)
'''
import kagglehub
import os
if __name__ == "__main__":
    #Set KAGGLEHUB_CACHE environment variable
    os.environ["KAGGLEHUB_CACHE"] = "data/"

    #Download latest version
    path = kagglehub.dataset_download("msambare/fer2013")

    print("Path to dataset files:", path)
