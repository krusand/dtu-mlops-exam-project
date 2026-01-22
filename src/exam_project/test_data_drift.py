from io import BytesIO
import json
import os
import tempfile
from typing import Callable, Iterable, List, Set, Tuple

from datetime import datetime, timedelta, timezone
from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluate import get_predictions
from model import BaseANN, BaseCNN, ViTClassifier


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'
MODELS = [("ann", BaseANN),
          ("cnn", BaseCNN),
          ("vit", ViTClassifier),
          ]

def extract_features(images):
    """Extract basic image features from a set of images."""
    # Convert PyTorch tensors to NumPy
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    
    features = []
    for img in images:
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        features.append([avg_brightness, contrast])
    return np.array(features)


def filter_date_blobs(date_blobs: Iterable[Blob], prefix: str, last_n_days: int) -> Set[str]:
    """Filters a series of date folders (blobs) in a GCP bucket based on last n days."""
    # Filtering date blobs
    valid_date_folders = set()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=last_n_days)
    prefix = prefix.rstrip('/') + '/'

    for date_blob in date_blobs:
        # Remove prefix and split path
        relative_path = date_blob.name.replace(prefix, "")
        parts = relative_path.split("/")

        if len(parts) < 1:
            continue

        date_str = parts[0]  # date expected in format "dd-mm-yyyy"

        try:
            folder_date = datetime.strptime(date_str, "%d-%m-%Y")
            # Making it timezone-aware in UTC so it can be compared to the cutoff_date
            folder_date = folder_date.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        if folder_date >= cutoff_date:
            valid_date_folders.add(date_str)

    return valid_date_folders


def load_labels_from_json(blob: Blob) -> dict[str, str]:
    """Returns mapping: image_name -> user_label"""
    data = json.loads(blob.download_as_text())

    return {
        entry["image_name"]: entry["user_label"]
        for entry in data
        if "image_name" in entry and "user_label" in entry
    }


def load_images_and_labels_from_date_folder(bucket: Bucket,
                                            date_folder: str,
                                            prefix: str,
                                            transform: Callable[[Image.Image], Tensor],
                                            class_to_idx: dict[str, int]
                                            ) -> Tuple[List[Tensor], List[int]]:
    images = []
    labels = []
    blobs = list(bucket.list_blobs(prefix=f"{prefix}{date_folder}/"))

    # Finding the JSON blob
    json_blob = next((b for b in blobs if b.name.endswith(".json")), None)

    if json_blob is None:
        raise RuntimeError(f"No JSON found in {date_folder}")

    label_lookup = load_labels_from_json(json_blob)

    # Loading images
    for blob in blobs:
        if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_name = blob.name.split("/")[-1]

        if image_name not in label_lookup:
            # Skip images without labels
            continue

        image_bytes = blob.download_as_bytes()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = transform(img)

        images.append(img)
        labels.append(class_to_idx[label_lookup[image_name]])

    return images, labels


def extract_class_mapping_from_json(blob: Blob) -> dict[str, int]:
    data = json.loads(blob.download_as_text())

    if not data:
        raise ValueError("Empty JSON")

    probabilities = data[0]["model_output"]["probabilities"]

    return {class_name: idx for idx, class_name in enumerate(sorted(probabilities.keys()))}


@hydra.main(config_path="configs", config_name="data", version_base=None)
def main(cfg):
    # Tensor transformation for image datasets
    transform = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor(),]
         )

    # Paths
    original_cwd = get_original_cwd()
    fer_processed_dir = os.path.join(original_cwd, cfg.paths.data_root, cfg.paths.processed_str)

    # Initialize GCP storage client
    storage_client = storage.Client(project="decent-seeker-484209-j2")
    bucket = storage_client.bucket("dtu-mlops-exam-project-data")

    # Obtaining requests blobs
    prefix = cfg.paths.requests_path
    date_blobs = bucket.list_blobs(prefix=prefix)
    print(f"Date blobs: {date_blobs}")
    # Filter date_blobs
    valid_date_blobs = filter_date_blobs(date_blobs=date_blobs, prefix=prefix, last_n_days=14)
    print(f"Validat date blobs: {valid_date_blobs}")
    # Retrieving json file from date blob of today
    first_date = datetime.now(timezone.utc).strftime("%d-%m-%Y")

    json_blob = next(
        b for b in bucket.list_blobs(prefix=f"{prefix}/{first_date}/")
        if b.name.endswith(".json")
    )

    # If no images has been uploaded today
    if json_blob is None:
        raise FileNotFoundError(f"No JSON found for today ({first_date})")

    # Dictionary to map labels to indices
    class_to_idx = extract_class_mapping_from_json(json_blob)
    print(f"Class dict: {class_to_idx}")
    # Load date from all valid date blobs (folders)
    ref_images = []
    ref_labels = []

    for date_folder in valid_date_blobs:
        imgs, lbls = load_images_and_labels_from_date_folder(
            bucket,
            date_folder,
            prefix,
            transform,
            class_to_idx
        )
        ref_images.extend(imgs)
        ref_labels.extend(lbls)
    print(ref_images)
    print(ref_labels)
    # Stacking images and labels into tensors
    ref_image_tensor = torch.stack(ref_images)           # Shape: [N, C, H, W]
    ref_label_tensor = torch.tensor(ref_labels)          # Shape: [N]
    ref_test_set = torch.utils.data.TensorDataset(ref_image_tensor, ref_label_tensor)

    # Extracting features from ref_image_tensor
    ref_features = extract_features(ref_image_tensor)

    # Loading fer_image_tensor and labels and extracting features
    fer_image_tensor = torch.load(f"{fer_processed_dir}/test_images.pt")
    fer_label_tensor = torch.load(f"{fer_processed_dir}/test_target.pt")
    fer_test_set = torch.utils.data.TensorDataset(fer_image_tensor, fer_label_tensor)
    fer_images_np = fer_image_tensor.numpy()
    fer_features = extract_features(fer_images_np)

    # List of features
    feature_columns = ["Average Brightness", "Contrast"]

    # Constructing separate dataframes
    ref_df = np.column_stack((ref_features, ["ref"] * ref_features.shape[0]))
    fer_df = np.column_stack((fer_features, ["fer"] * fer_features.shape[0]))

    # Combining features
    combined_features = np.vstack((ref_df, fer_df))
    feature_df = pd.DataFrame(combined_features, columns=feature_columns + ["Dataset"])
    feature_df[feature_columns] = feature_df[feature_columns].astype(float)

    # Final dataframes for evidently
    reference_data = feature_df[feature_df["Dataset"] == "ref"].drop(columns=["Dataset"])
    current_data = feature_df[feature_df["Dataset"] == "fer"].drop(columns=["Dataset"])
    
    # Today's date to be added as suffix to exported files
    today_str = datetime.now(timezone.utc).strftime("%d%m%y")

    # Generating data drift report
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    os.makedirs("reports", exist_ok=True)
    report.save_html(f"reports/data_drift_{today_str}.html")

    model_names = []
    curr_accs = []
    curr_f1s = []
    curr_n_samples = []
    ref_accs = []
    ref_f1s = []
    ref_n_samples = []
    
    # Iterate over models
    for model_name, model_class in tqdm(MODELS):
        # Find model checkpoints in models/cnn folder
        blobs = list(bucket.list_blobs(prefix=f"models/{model_name}"))
        ckpt_files = [blob for blob in blobs if blob.name.endswith(".ckpt")]

        if not ckpt_files:
            continue

        ckpt_blob = ckpt_files[0]

        tmp_dir = tempfile.mkdtemp()
        ckpt_path = os.path.join(tmp_dir, f"{model_name}.ckpt")

        ckpt_blob.download_to_filename(ckpt_path)

        model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
        model.eval()
        
        # Converting ref to grayscale for inference
        ref_image_tensor_gray = ref_image_tensor.mean(dim=1, keepdim=True)
        ref_test_set = torch.utils.data.TensorDataset(ref_image_tensor_gray, ref_label_tensor)

        # Dataloaders
        ref_test_loader = DataLoader(ref_test_set, persistent_workers=True, num_workers=9)
        fer_test_loader = DataLoader(fer_test_set, persistent_workers=True, num_workers=9, shuffle=True)

        # Get predictions
        n_ref = len(ref_test_set)
        y_pred_ref, y_true_ref = get_predictions(model, ref_test_loader, DEVICE, n_ref)
        n_fer = 100
        y_pred_fer, y_true_fer = get_predictions(model, fer_test_loader, DEVICE, n_fer)

        # Compute evaluation metrics
        ref_acc = accuracy_score(y_true_ref, y_pred_ref)
        ref_f1 = f1_score(y_true_ref, y_pred_ref, average="weighted")
        fer_acc = accuracy_score(y_true_fer, y_pred_fer)
        fer_f1 = f1_score(y_true_fer, y_pred_fer, average="weighted")

        # Appending variables to lists
        model_names.append(model_name)
        curr_accs.append(fer_acc)
        curr_f1s.append(fer_f1)
        curr_n_samples.append(n_fer)
        ref_accs.append(ref_acc)
        ref_f1s.append(ref_f1)
        ref_n_samples.append(n_ref)

    # Saving data as csv
    data_dict = {"model": model_names,
                 "curr_acc": curr_accs,
                 "curr_f1": curr_f1s,
                 "curr_n_samples": curr_n_samples,
                 "ref_acc": ref_accs,
                 "ref_f1": ref_f1s,
                 "ref_n_samples": ref_n_samples}
    df = pd.DataFrame(data_dict)
    df.to_csv(f"reports/performance_comparison_{today_str}.csv")


if __name__ == "__main__":
    main()


