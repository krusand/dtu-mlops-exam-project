from io import BytesIO
import os
import tempfile

from evidently.legacy.metrics import DataDriftTable
from evidently.legacy.report import Report
from google.cloud import storage
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
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

    # Looking inside MMA blobs on GCP
    prefix = cfg.paths.mma_path
    blobs = bucket.list_blobs(prefix=prefix)

    # Collect unique class folder names
    class_names = set()
    for blob in blobs:
        parts = blob.name.replace(prefix, "").split("/")
        if len(parts) > 1:  # ensures it's inside a class folder
            class_names.add(parts[0])

    # Mapping class names to integer labels
    class_names = sorted(list(class_names))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    # Looping through subfolders
    ref_images = []
    ref_labels = []
    for cls_name in class_names:
        cls_folder = f"{prefix}{cls_name}/"
        blobs = bucket.list_blobs(prefix=cls_folder)
        
        for blob in blobs:
            if blob.name.endswith((".jpg", ".png", ".jpeg")):
                image_bytes = blob.download_as_bytes()
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
                img = transform(img)
                ref_images.append(img)
                ref_labels.append(class_to_idx[cls_name])

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
 
    # Generating data drift report
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("reports/data_drift.html")

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
            raise FileNotFoundError("No .ckpt files found in the bucket!")

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
        n_ref = len(ref_test_loader)
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
    df.to_csv("reports/performance_comparison.csv")


if __name__ == "__main__":
    main()


