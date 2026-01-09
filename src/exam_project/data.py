import kagglehub
import os
from pathlib import Path
import torch
'''
import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set
'''
root = 'data/'
raw_dir = os.path.join(root, 'raw/datasets/msambare/fer2013/versions/1/')
processed_dir = os.path.join(root, 'processed')

def create_data_folders(raw_dir, processed_dir):
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder

# Function to open images
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')# grayscale==1 channel

# Transform: convert to tensor and optionally normalize
def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),  # converts PIL image to tensor with values [0,1]
    ])
    return transform

def get_dataset(root, transform):
    dataset = DatasetFolder(
        root=root,  # replace with your folder path
        loader=pil_loader,
        extensions=['jpg'],
        transform=transform
    )
    return dataset

def get_image_labels_tensors(dataset):
    images_list = []
    labels_list = []

    for img, label in dataset:
        images_list.append(img)        # img is already a tensor from transform
        labels_list.append(label)      # label is an integer

    # Stack into single tensors
    images_tensor = torch.stack(images_list)   # shape: [num_images, 3, H, W]
    labels_tensor = torch.tensor(labels_list)  # shape: [num_images]
    return images_tensor, labels_tensor

def save_image_labels(images, labels, processed_dir, traintest):
    torch.save(images,os.path.join(processed_dir,f"{traintest}_images.pt"))
    torch.save(labels,os.path.join(processed_dir,f"{traintest}_target.pt"))

def save_metadata(metadata, processed_dir):
    torch.save(metadata, os.path.join(processed_dir,'class_to_idx.pt'))

def get_split_index(N, frac=0.8):
    split = int(frac * N)  # 80%
    indices = torch.randperm(N)  # shuffled indices
    train_idx = indices[:split]
    val_idx   = indices[split:]
    return train_idx, val_idx

def save_data(train_dataset, test_dataset, processed_dir):
    #Split full training set into training and validation set
    train_images_all, train_labels_all = get_image_labels_tensors(train_dataset)
    train_idx, val_idx = get_split_index(train_images_all.size(0), 0.8)
    #Save training and validation sets
    save_image_labels(train_images_all[train_idx], train_labels_all[train_idx], processed_dir, 'train')
    save_image_labels(train_images_all[val_idx], train_labels_all[val_idx], processed_dir, 'val')
    #Save test set
    save_image_labels(*get_image_labels_tensors(test_dataset), processed_dir, 'test')

def preprocess_data(raw_dir, processed_dir):
    #Create data folders if they don't already exist
    create_data_folders(raw_dir, processed_dir)

    #Get transform and load data/raw/
    transform = get_transform()
    train_dataset = get_dataset(os.path.join(raw_dir,'train'), transform)
    test_dataset  = get_dataset(os.path.join(raw_dir,'test'), transform)

    #Save datasets in data/processed
    print ('Converting datasets .pt files...')
    save_data(train_dataset, test_dataset, processed_dir)
    save_metadata(train_dataset.class_to_idx, processed_dir)
    print ('Done.')

def load_metadata(processed_dir):
    return torch.load(os.path.join(processed_dir,'class_to_idx.pt'))

def load_data(processed_dir) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
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

if __name__ == "__main__":

    #Set KAGGLEHUB_CACHE environment variable
    os.environ["KAGGLEHUB_CACHE"] = os.path.join(root, 'raw')

    #Download latest version
    path = kagglehub.dataset_download("msambare/fer2013")
    print("Path to dataset files:", path)

    #Load datasets from data/raw and save in data/preprocessed    
    preprocess_data(raw_dir, processed_dir)

    #Load datsets from data/processed
    train_set, val_set, test_set = load_data(processed_dir)