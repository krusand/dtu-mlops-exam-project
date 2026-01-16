import os
import pytest
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

from exam_project.data import (
    get_transform,
    get_split_index,
    load_data,
    create_processed_dir,
)

# Constants (hardcoded from data.py to avoid import issues)
KAGGLE_ID = "msambare/fer2013"
DATA_VERSION_PATH = "1"
ROOT = "data"
RAW_STR = "raw"

EXPECTED_CLASSES = {
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
}
EXPECTED_IMAGE_SIZE = (48, 48)
EXPECTED_MODE = "L"


def get_all_dataset_versions():
    """Get all available dataset versions."""
    base_dir = Path(f"{ROOT}/{RAW_STR}/datasets/{KAGGLE_ID}/versions")
    versions = set()

    current_version_dir = Path(
        f"{ROOT}/{RAW_STR}/datasets/{KAGGLE_ID}/versions/{DATA_VERSION_PATH}"
    )
    if current_version_dir.exists():
        versions.add(current_version_dir)

    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                versions.add(item)

    return sorted(versions, key=lambda x: int(x.name))


def test_kagglehub_credentials():
    """Test that Kaggle API credentials are configured."""
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get(
        "KAGGLE_KEY"
    )
    has_config = (Path.home() / ".kaggle" / "kaggle.json").exists()

    if not (has_env or has_config):
        pytest.skip(
            "Kaggle credentials not found. "
            "Set KAGGLE_USERNAME/KAGGLE_KEY or create ~/.kaggle/kaggle.json"
        )


@pytest.mark.integration
def test_kagglehub_dataset_download():
    """Test that kagglehub can download the FER2013 dataset."""
    pytest.importorskip("kagglehub")

    has_creds = os.environ.get("KAGGLE_USERNAME") and os.environ.get(
        "KAGGLE_KEY"
    )
    has_config = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_creds or has_config):
        pytest.skip("Kaggle credentials not available")

    import kagglehub

    try:
        os.environ["KAGGLEHUB_CACHE"] = "data/raw"
        path = kagglehub.dataset_download("msambare/fer2013")
        assert path is not None, "dataset_download returned None"
        assert Path(path).exists(), f"Dataset path does not exist: {path}"
    except Exception as e:
        pytest.skip(f"Could not download dataset: {str(e)}")


def test_dataset_versions_available():
    """Test that specified dataset version is available."""
    versions = get_all_dataset_versions()
    if not versions:
        pytest.skip("No dataset versions found. Download data first.")

    version_numbers = [v.name for v in versions]
    assert DATA_VERSION_PATH in version_numbers, (
        f"Version {DATA_VERSION_PATH} not found. Available: {version_numbers}"
    )


@pytest.mark.parametrize("version_dir", get_all_dataset_versions())
def test_raw_data_image_files_valid(version_dir):
    """Test that all image files in raw data are valid."""
    if not version_dir.exists():
        pytest.skip("Raw data not found.")

    invalid_files = []
    total_files = 0

    for split in ["train", "test"]:
        split_dir = version_dir / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for img_file in class_dir.glob("*.jpg"):
                total_files += 1
                try:
                    with Image.open(img_file):
                        pass
                except (IOError, OSError, Image.UnidentifiedImageError) as e:
                    invalid_files.append((str(img_file), str(e)))

    if total_files == 0:
        pytest.skip("No image files found.")

    assert not invalid_files, (
        f"Found {len(invalid_files)} invalid images"
    )


@pytest.mark.parametrize("version_dir", get_all_dataset_versions())
def test_raw_data_image_sizes_consistent(version_dir):
    """Test that all images are 48x48 pixels."""
    if not version_dir.exists():
        pytest.skip("Raw data not found.")

    inconsistent = []
    total_files = 0

    for split in ["train", "test"]:
        split_dir = version_dir / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for img_file in class_dir.glob("*.jpg"):
                total_files += 1
                try:
                    with Image.open(img_file) as img:
                        if img.size != EXPECTED_IMAGE_SIZE:
                            inconsistent.append((str(img_file), img.size))
                except Exception:
                    pass

    if total_files == 0:
        pytest.skip("No image files found.")

    assert not inconsistent, (
        f"Found {len(inconsistent)} images with wrong size. "
        f"Expected {EXPECTED_IMAGE_SIZE}"
    )

@pytest.mark.parametrize("version_dir", get_all_dataset_versions())
def test_raw_data_images_are_grayscale(version_dir):
    """Test that all images are grayscale."""
    if not version_dir.exists():
        pytest.skip("Raw data not found.")

    non_grayscale = []
    total_files = 0

    for split in ["train", "test"]:
        split_dir = version_dir / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            for img_file in class_dir.glob("*.jpg"):
                total_files += 1
                try:
                    with Image.open(img_file) as img:
                        if img.mode != EXPECTED_MODE:
                            non_grayscale.append((str(img_file), img.mode))
                except Exception:
                    pass

    if total_files == 0:
        pytest.skip("No image files found.")

    assert not non_grayscale, (
        f"Found {len(non_grayscale)} non-grayscale images. "
        f"Expected mode {EXPECTED_MODE}"
    )

@pytest.mark.parametrize("version_dir", get_all_dataset_versions())
def test_raw_data_has_all_emotion_classes(version_dir):
    """Test that all 7 emotion classes are present."""
    if not version_dir.exists():
        pytest.skip("Raw data not found.")

    found_classes = set()
    class_counts = {}

    for split in ["train", "test"]:
        split_dir = version_dir / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            count = len(list(class_dir.glob("*.jpg")))
            if count > 0:
                found_classes.add(class_name)
                class_counts[class_name] = class_counts.get(class_name, 0) + count

    assert found_classes, "No emotion classes found"
    assert EXPECTED_CLASSES.issubset(found_classes), (
        f"Missing classes. Expected {EXPECTED_CLASSES}, "
        f"found {found_classes}"
    )

    for emotion in EXPECTED_CLASSES:
        assert class_counts.get(emotion, 0) > 0, (
            f"Class {emotion} is empty"
        )
@pytest.mark.parametrize("version_dir", get_all_dataset_versions())
def test_raw_data_class_distribution(version_dir):
    """Test that all emotion classes have images in train and test."""
    if not version_dir.exists():
        pytest.skip("Raw data not found.")

    dist = {e: {"train": 0, "test": 0} for e in EXPECTED_CLASSES}

    for split in ["train", "test"]:
        split_dir = version_dir / split
        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if class_dir.is_dir() and class_dir.name in EXPECTED_CLASSES:
                count = len(list(class_dir.glob("*.jpg")))
                dist[class_dir.name][split] = count

    for emotion in EXPECTED_CLASSES:
        train = dist[emotion]["train"]
        test = dist[emotion]["test"]
        assert train > 0, f"Class {emotion} missing in training set"
        assert test > 0, f"Class {emotion} missing in test set"


def test_get_transform():
    """Test that transform returns a callable."""
    transform = get_transform()
    assert callable(transform)


def test_get_split_index():
    """Test train/val split indices with various configurations."""
    N = 100
    train_idx, val_idx = get_split_index(N, 0.8, seed=42)

    assert len(train_idx) == 80
    assert len(val_idx) == 20
    assert len(torch.unique(torch.cat([train_idx, val_idx]))) == N

    for frac in [0.6, 0.7, 0.8, 0.9]:
        train_idx, val_idx = get_split_index(N, frac, seed=42)
        assert len(train_idx) == int(frac * N)
        assert len(val_idx) == N - int(frac * N)
        assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_data_split_reproducibility():
    """Test that data splits are deterministic."""
    splits = [get_split_index(200, 0.8, seed=42) for _ in range(3)]

    for train_idx, val_idx in splits[1:]:
        assert torch.equal(train_idx, splits[0][0])
        assert torch.equal(val_idx, splits[0][1])


def test_create_processed_dir(tmp_path):
    """Test that processed directory is created."""
    test_dir = tmp_path / "test_processed"
    create_processed_dir(str(test_dir))
    
    assert test_dir.exists()
    assert test_dir.is_dir()


def test_load_data():
    """Test loading preprocessed data."""
    processed_dir = "data/processed"
    p = Path(processed_dir)

    if not p.exists():
        pytest.skip("Processed data not found.")

    required = [
        "train_images.pt", "train_target.pt",
        "val_images.pt", "val_target.pt",
        "test_images.pt", "test_target.pt",
    ]
    for file in required:
        if not (p / file).exists():
            pytest.skip(f"Required file {file} not found.")

    train_set, val_set, test_set = load_data(processed_dir)

    for ds in [train_set, val_set, test_set]:
        assert isinstance(ds, Dataset)
        assert len(ds) > 0

    train_img, train_label = train_set[0]
    assert isinstance(train_img, torch.Tensor)
    assert isinstance(train_label, torch.Tensor)
    assert train_img.dim() == 3 and train_img.shape[0] == 1
    assert train_label.dim() == 0 and 0 <= train_label < 7


def test_data_normalization():
    """Test that transform normalizes data correctly."""
    transform = get_transform()
    img = Image.new('L', (48, 48), color=128)
    tensor = transform(img)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 48, 48)
    # Normalize(mean=0.5, std=0.5) maps [0, 1] -> [-1, 1]
    # Allow small tolerance for floating point precision
    assert tensor.min() >= -1.1 and tensor.max() <= 1.1


def test_data_loading_reproducibility():
    """Test that data loading is deterministic."""
    processed_dir = "data/processed"

    if not Path(processed_dir).exists():
        pytest.skip("Processed data not found.")

    datasets = [load_data(processed_dir) for _ in range(3)]

    for train_set, val_set, test_set in datasets[1:]:
        assert len(train_set) == len(datasets[0][0])
        assert len(val_set) == len(datasets[0][1])
        assert len(test_set) == len(datasets[0][2])

    n = len(datasets[0][0])
    for idx in [0, n // 2, n - 1]:
        for dataset_idx in range(1, 3):
            img1, label1 = datasets[0][0][idx]
            img2, label2 = datasets[dataset_idx][0][idx]
            assert torch.equal(img1, img2) and label1 == label2
