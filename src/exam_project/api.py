import io
import logging
import os
import tempfile
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import storage
from PIL import Image
from torchvision import transforms

from exam_project.model import BaseANN, BaseCNN, ViTClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Classifier API",
    description="Classifies emotions from facial images",
    version="1.0.0"
)

model = None
device = None
current_model_name = None
loaded_model_path = None

# Cloud configuration
USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "emotion-classifier-models")


def get_available_models():
    """Get available models from local filesystem or GCS bucket."""
    available = {}

    if USE_GCS:
        # Load models from GCS bucket
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCS_BUCKET_NAME)

            for model_type in ["cnn", "ann", "vit"]:
                prefix = f"models/{model_type}/"
                blobs = bucket.list_blobs(prefix=prefix)

                for blob in blobs:
                    if blob.name.endswith(".ckpt"):
                        try:
                            loss = float(blob.name.split('=')[-1].replace('.ckpt', ''))
                        except (ValueError, IndexError):
                            loss = None

                        available[model_type] = {
                            "path": f"gs://{GCS_BUCKET_NAME}/{blob.name}",
                            "validation_loss": loss,
                            "filename": blob.name.split('/')[-1]
                        }
                        break  # Use first .ckpt file for this model type

            logger.info(f"Loaded models from GCS: {list(available.keys())}")
        except Exception as e:
            logger.error(f"Error loading models from GCS: {e}", exc_info=True)
    else:
        # Load models from local filesystem
        models_dir = Path("models")
        if models_dir.exists():
            for model_name in ["cnn", "ann", "vit"]:
                model_subdir = models_dir / model_name
                if model_subdir.is_dir():
                    ckpt_files = list(model_subdir.glob("*.ckpt"))
                    if ckpt_files:
                        ckpt_file = ckpt_files[0]  # Get first .ckpt file
                        try:
                            loss = float(str(ckpt_file).split('=')[-1].replace('.ckpt', ''))
                            available[model_name] = {
                                "path": str(ckpt_file),
                                "validation_loss": loss,
                                "filename": ckpt_file.name
                            }
                        except ValueError:
                            available[model_name] = {
                                "path": str(ckpt_file),
                                "validation_loss": None,
                                "filename": ckpt_file.name
                            }

        logger.info(f"Loaded models from local filesystem: {list(available.keys())}")

    return available




def load_model_checkpoint(model_name: Optional[str] = None):
    """Load model checkpoint from local filesystem or GCS bucket."""
    global model, device, current_model_name, loaded_model_path

    if model is not None and (model_name is None or model_name == current_model_name):
        return

    logger.info(f"Loading model (model_name={model_name})...")

    try:
        device = torch.device("cpu")

        # Get available models
        available = get_available_models()

        if model_name:
            if model_name not in available:
                msg = f"Model '{model_name}' not found. Available: {list(available.keys())}"
                raise ValueError(msg)
            checkpoint_path = available[model_name]["path"]
            current_model_name = model_name
        else:
            # Try to use first available model
            if available:
                current_model_name = list(available.keys())[0]
                checkpoint_path = available[current_model_name]["path"]
            else:
                raise FileNotFoundError("No models found in models directory or GCS bucket")

        # Map model name to correct class
        model_class_map = {
            "cnn": BaseCNN,
            "ann": BaseANN,
            "vit": ViTClassifier
        }
        model_class = model_class_map.get(current_model_name, BaseCNN)

        # Load from GCS if path starts with gs://
        if checkpoint_path.startswith("gs://"):
            bucket_name = checkpoint_path.split("/")[2]
            blob_path = "/".join(checkpoint_path.split("/")[3:])

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp:
                blob.download_to_file(tmp)
                tmp_path = tmp.name

            model = model_class.load_from_checkpoint(tmp_path, map_location=device)
            os.unlink(tmp_path)
            logger.info(f"Model loaded from GCS: {checkpoint_path}")
        else:
            # Load from local filesystem
            model = model_class.load_from_checkpoint(checkpoint_path, map_location=device)
            logger.info(f"Model loaded from local filesystem: {checkpoint_path}")

        loaded_model_path = checkpoint_path
        model.eval()
        logger.info(f"Model loaded successfully ({current_model_name})")

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise


def get_image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


@app.get("/")
async def root():
    load_model_checkpoint()
    return {
        "status": "ok",
        "service": "emotion-classifier",
        "model": current_model_name
    }


@app.get("/models/")
async def list_models():
    available = get_available_models()
    return {
        "available_models": available,
        "current_model": current_model_name,
        "total": len(available)
    }


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    accept: Optional[str] = Header(None)
):
    logger.info(f"Authorization header received: {authorization}")
    logger.info(f"Accept header received: {accept}")
    if model_name:
        logger.info(f"Model requested: {model_name}")

    if authorization != "dtu":
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    if accept != "application/json":
        raise HTTPException(status_code=400, detail="Invalid accept header")

    load_model_checkpoint(model_name=model_name)

    emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    try:
        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from file")

        if not contents:
            logger.error("File contents are empty")
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"error": "Empty file", "message": "Uploaded file is empty"}
            )

        try:
            logger.info("Attempting to open image with PIL...")
            image_data = Image.open(io.BytesIO(contents))
            logger.info(f"Image loaded: mode={image_data.mode}, size={image_data.size}")
        except Exception as e:
            logger.error(f"Failed to open image: {e}", exc_info=True)
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"error": "Invalid image format", "message": f"Could not open image: {str(e)}"}
            )

        if image_data.mode != 'L':
            image_data = image_data.convert('L')

        # Resize image to 48x48 for model
        image_data = image_data.resize((48, 48), Image.Resampling.LANCZOS)
        logger.info(f"Image resized to: {image_data.size}")

        transform = get_image_transform()
        image_tensor = transform(image_data)

        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        emotion = emotion_labels[predicted.item()]
        confidence_value = confidence.item()

        prob_dict = {
            emotion_labels[i]: probabilities[0, i].item()
            for i in range(len(emotion_labels))
        }

        return {
            "emotion": emotion,
            "confidence": confidence_value,
            "probabilities": prob_dict,
            "message": "Prediction successful"
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
