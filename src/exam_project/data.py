import os
from pathlib import Path
from typing import Callable, Dict, Literal

import hydra
import kagglehub
import PIL
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import DatasetFolder

#Define type hints
Transform = Callable[[Image.Image], torch.Tensor]
TrainValTestMode = Literal["train", "val", "test"]

def create_processed_dir(processed_dir:str)->None:
    """Create data/processed directory if it does not exist"""
    Path(processed_dir).mkdir(parents=True, exist_ok=True)


def pil_loader(path:str)->PIL.Image.Image:
    """Load images with PIL"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')# grayscale==1 channel


def get_transform()->Transform:
    """Transform PIL image to tensor, and normalise to range [-1,1]"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts PIL image to tensor with values [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5]), # convert range to [-1,1]
    ])
    return transform


def get_dataset(root: str, transform: Transform)->torch.utils.data.Dataset:
    """Get dataset using torchvision functionality"""
    dataset = DatasetFolder(
        root=root,
        loader=pil_loader,
        extensions=['jpg'],
        transform=transform
    )
    return dataset


def get_image_labels_tensors(dataset: torch.utils.data.Dataset)->tuple[torch.Tensor,torch.Tensor]:
    """Get images and labels tensors"""
    images_list = []
    labels_list = []

    for img, label in dataset:
        images_list.append(img)        # img is a tensor
        labels_list.append(label)      # label is an integer

    # Stack into single tensors
    images_tensor = torch.stack(images_list)   # shape: [num_images, 1, H, W]
    labels_tensor = torch.tensor(labels_list)  # shape: [num_images]
    return images_tensor, labels_tensor


def save_image_labels(images:torch.Tensor, labels: torch.Tensor, 
                      processed_dir:str, traintest:TrainValTestMode)->None:
    """Save images and labels tensors"""
    torch.save(images,os.path.join(processed_dir,f"{traintest}_images.pt"))
    torch.save(labels,os.path.join(processed_dir,f"{traintest}_target.pt"))


def save_metadata(metadata:Dict[str,int], processed_dir:str)->None:
    """Save metadata mapping string classes to class integers"""
    torch.save(metadata, os.path.join(processed_dir,'class_to_idx.pt'))


def get_split_index(N:int, frac:float, seed:int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get indices for splitting torch tensor using random shuffle"""
    split = int(frac * N)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(N, generator=g)

    train_idx = indices[:split]
    val_idx   = indices[split:]
    return train_idx, val_idx


def save_data(train_dataset:torch.utils.data.Dataset, test_dataset:torch.utils.data.Dataset, 
              processed_dir:str, trainvalsplit:float, seed:int)->None:
    """Save images and labels tensors for train, val and test, and metadata"""
    #Split full training set into training and validation set
    train_images_all, train_labels_all = get_image_labels_tensors(train_dataset)
    train_idx, val_idx = get_split_index(train_images_all.size(0), frac=trainvalsplit, seed=seed)
    #Save training and validation sets
    save_image_labels(train_images_all[train_idx], train_labels_all[train_idx], processed_dir, 'train')
    save_image_labels(train_images_all[val_idx], train_labels_all[val_idx], processed_dir, 'val')
    #Save test set
    save_image_labels(*get_image_labels_tensors(test_dataset), processed_dir, 'test')


def preprocess_data(raw_dir:str, processed_dir:str, trainvalsplit:float, seed:int)->None:
    """Load data from data/raw/ and save .pt files in data/preprocessed"""
    #Get transform and load data/raw/
    transform = get_transform()
    train_dataset = get_dataset(os.path.join(raw_dir,'train'), transform)
    test_dataset  = get_dataset(os.path.join(raw_dir,'test'), transform)

    #Save datasets in data/processed
    print ('Converting datasets .pt files...')
    save_data(train_dataset, test_dataset, processed_dir, trainvalsplit, seed)
    save_metadata(train_dataset.class_to_idx, processed_dir)
    print ('Done.')


def load_metadata(processed_dir:str)->torch.Tensor:
    """Load metadata tensor"""
    return torch.load(os.path.join(processed_dir,'class_to_idx.pt'))


def load_data(processed_dir:str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train, val and test datasets for FER2013."""
    train_images = torch.load(os.path.join(processed_dir,"train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir,"train_target.pt"))
    val_images = torch.load(os.path.join(processed_dir,"val_images.pt"))
    val_target = torch.load(os.path.join(processed_dir,"val_target.pt"))
    test_images = torch.load(os.path.join(processed_dir,"test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir,"test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    val_set = torch.utils.data.TensorDataset(val_images, val_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, val_set, test_set

@hydra.main(config_path="../../configs/", config_name="data", version_base=None)
def main(cfg):
    #Set KAGGLEHUB_CACHE environment variable
    os.environ["KAGGLEHUB_CACHE"] = os.path.join(cfg.paths.data_root, cfg.paths.raw_str)

    #Download latest version of data from kaggle
    path = kagglehub.dataset_download(cfg.paths.kaggle_id)
    print("Path to dataset files:", path)

    #Define directories
    raw_dir = os.path.join(cfg.paths.data_root, f'{cfg.paths.raw_str}/datasets/{cfg.paths.kaggle_id}/versions/{cfg.paths.data_version_path}/')
    processed_dir = os.path.join(cfg.paths.data_root, cfg.paths.processed_str)

    #Create processed dir if it doesn't already exist
    create_processed_dir(processed_dir)

    #Assert folders exist
    assert(os.path.exists(raw_dir))
    assert(os.path.exists(processed_dir))

    #Load datasets from data/raw and save .pt images and labels in data/preprocessed    
    preprocess_data(raw_dir, processed_dir, cfg.hyperparameters.trainvalsplit, seed=cfg.hyperparameters.seed)

    #Load datsets from data/processed
    train_set, val_set, test_set = load_data(processed_dir)

if __name__ == "__main__":
    main()