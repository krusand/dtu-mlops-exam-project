import kagglehub
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    """Dataset class for loading processed data."""
    
    def __init__(self, data_path: str):
        """Initialize dataset.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        return None


def load_images_from_folder(folder_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load images and labels from a folder structure.
    
    The folder should contain subdirectories named after emotion classes,
    with images inside each subdirectory.
    
    Args:
        folder_path: Path to the folder containing emotion subdirectories
        
    Returns:
        Tuple of (images tensor, labels tensor)
    """
    images = []
    labels = []
    
    # Map emotion folder names to numeric labels
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'neutral': 4,
        'sad': 5,
        'surprise': 6
    }
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Iterate through emotion subdirectories
    for emotion_name, label in emotion_map.items():
        emotion_path = folder_path / emotion_name
        if not emotion_path.exists():
            continue
            
        # Load all images from this emotion folder
        for img_file in emotion_path.glob('*.jpg'):
            try:
                # Load image and convert to grayscale if needed
                img = Image.open(img_file).convert('L')
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
    
    # Convert to tensors
    if len(images) == 0:
        raise ValueError(f"No images found in {folder_path}")
    
    images_tensor = torch.from_numpy(np.array(images))
    labels_tensor = torch.from_numpy(np.array(labels, dtype=np.int64))
    
    return images_tensor, labels_tensor


def standardize_images(images: torch.Tensor) -> torch.Tensor:
    """Standardize images to have zero mean and unit variance.
    
    Args:
        images: Input images tensor
        
    Returns:
        Standardized images tensor
    """
    mean = images.mean()
    std = images.std()
    
    # Avoid division by zero
    if std < 1e-8:
        std = 1.0
    
    standardized = (images - mean) / std
    return standardized


def process_data(raw_path: str, processed_path: str, val_split: float = 0.15):
    """Process raw data into train/val/test splits.
    
    Args:
        raw_path: Path to raw data directory
        processed_path: Path to save processed data
        val_split: Fraction of training data to use for validation
    """
    raw_path = Path(raw_path)
    processed_path = Path(processed_path)
    
    # Create processed directory if it doesn't exist
    processed_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading training data...")
    train_folder = raw_path / "train"
    train_images, train_labels = load_images_from_folder(train_folder)
    
    print("Loading test data...")
    test_folder = raw_path / "test"
    test_images, test_labels = load_images_from_folder(test_folder)
    
    # Split training data into train and validation
    n_train = len(train_images)
    n_val = int(n_train * val_split)
    
    # Shuffle indices
    indices = torch.randperm(n_train)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Split the data
    val_images = train_images[val_indices]
    val_labels = train_labels[val_indices]
    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    
    print(f"Data splits - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Standardize images (compute statistics on training set only)
    print("Standardizing images...")
    train_mean = train_images.mean()
    train_std = train_images.std()
    
    if train_std < 1e-8:
        train_std = 1.0
    
    train_images = (train_images - train_mean) / train_std
    val_images = (val_images - train_mean) / train_std
    test_images = (test_images - train_mean) / train_std
    
    # Save processed data
    print("Saving processed data...")
    torch.save(train_images, processed_path / "train_images.pt")
    torch.save(train_labels, processed_path / "train_targets.pt")
    torch.save(val_images, processed_path / "val_images.pt")
    torch.save(val_labels, processed_path / "val_targets.pt")
    torch.save(test_images, processed_path / "test_images.pt")
    torch.save(test_labels, processed_path / "test_targets.pt")
    
    print(f"Processed data saved to {processed_path}")
    print(f"Train images shape: {train_images.shape}")
    print(f"Val images shape: {val_images.shape}")
    print(f"Test images shape: {test_images.shape}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Process data mode
        raw_path = sys.argv[1]
        processed_path = sys.argv[2]
        process_data(raw_path, processed_path)
    else:
        # Download data mode (backward compatibility)
        os.environ["KAGGLEHUB_CACHE"] = "data/raw/"
        path = kagglehub.dataset_download("msambare/fer2013")
        print("Path to dataset files:", path)
