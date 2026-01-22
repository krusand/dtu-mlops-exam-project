import io
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google.api_core.exceptions import Forbidden, NotFound, PreconditionFailed
from google.cloud import storage
from PIL import Image
from torchvision import transforms
from zoneinfo import ZoneInfo 


from exam_project.model import BaseANN, BaseCNN, ViTClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Emotion Classifier API",
    description="Classifies emotions from facial images",
    version="1.0.0",
)

model = None
device = None
current_model_name = None
loaded_model_path = None


USE_GCS = os.getenv("USE_GCS", "false").lower() == "true"
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "emotion-classifier-models")

SAVE_REQUESTS = os.getenv("SAVE_REQUESTS", "true").lower() == "true"
REQUESTS_BUCKET_NAME = os.getenv("REQUESTS_BUCKET_NAME", "dtu-mlops-exam-project-data")
REQUESTS_PREFIX = os.getenv("REQUESTS_PREFIX", "requests")
FOLDER_TZ = os.getenv("FOLDER_TZ", "Europe/Copenhagen")
DAILY_METADATA_FILENAME = os.getenv("DAILY_METADATA_FILENAME", "metadata.json")

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_storage_client: Optional[storage.Client] = None


def get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def get_folder_date_str() -> str:
    """Folder name format: DD-MM-YYYY based on Europe/Copenhagen (or FOLDER_TZ)."""
    if ZoneInfo is not None:
        now = datetime.now(ZoneInfo(FOLDER_TZ))
    else:
        now = datetime.now(timezone.utc)
    return now.strftime("%d-%m-%Y")


def get_available_models():
    """Get available models from local filesystem or GCS bucket."""
    available = {}

    if USE_GCS:
        try:
            storage_client = get_storage_client()
            bucket = storage_client.bucket(GCS_BUCKET_NAME)

            for model_type in ["cnn", "ann", "vit"]:
                prefix = f"models/{model_type}/"
                blobs = bucket.list_blobs(prefix=prefix)

                for blob in blobs:
                    if blob.name.endswith(".ckpt"):
                        try:
                            loss = float(blob.name.split("=")[-1].replace(".ckpt", ""))
                        except (ValueError, IndexError):
                            loss = None

                        available[model_type] = {
                            "path": f"gs://{GCS_BUCKET_NAME}/{blob.name}",
                            "validation_loss": loss,
                            "filename": blob.name.split("/")[-1],
                        }
                        break
            logger.info(f"Loaded models from GCS: {list(available.keys())}")
        except Exception as e:
            logger.error(f"Error loading models from GCS: {e}", exc_info=True)
    else:
        models_dir = Path("models")
        if models_dir.exists():
            for model_name in ["cnn", "ann", "vit"]:
                model_subdir = models_dir / model_name
                if model_subdir.is_dir():
                    ckpt_files = list(model_subdir.glob("*.ckpt"))
                    if ckpt_files:
                        ckpt_file = ckpt_files[0]
                        try:
                            loss = float(str(ckpt_file).split("=")[-1].replace(".ckpt", ""))
                            available[model_name] = {
                                "path": str(ckpt_file),
                                "validation_loss": loss,
                                "filename": ckpt_file.name,
                            }
                        except ValueError:
                            available[model_name] = {
                                "path": str(ckpt_file),
                                "validation_loss": None,
                                "filename": ckpt_file.name,
                            }

        logger.info(f"Loaded models from local filesystem: {list(available.keys())}")

    return available


def load_model_checkpoint(model_name: Optional[str] = None):
    """Load model checkpoint from local filesystem or GCS bucket."""
    global model, device, current_model_name, loaded_model_path

    if model is not None and (model_name is None or model_name == current_model_name):
        return

    logger.info(f"Loading model (model_name={model_name})...")

    device = torch.device("cpu")
    available = get_available_models()

    if model_name:
        if model_name not in available:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(available.keys())}")
        checkpoint_path = available[model_name]["path"]
        current_model_name = model_name
    else:
        if available:
            current_model_name = list(available.keys())[0]
            checkpoint_path = available[current_model_name]["path"]
        else:
            raise FileNotFoundError("No models found in models directory or GCS bucket")

    model_class_map = {"cnn": BaseCNN, "ann": BaseANN, "vit": ViTClassifier}
    model_class = model_class_map.get(current_model_name, BaseCNN)

    if checkpoint_path.startswith("gs://"):
        bucket_name = checkpoint_path.split("/")[2]
        blob_path = "/".join(checkpoint_path.split("/")[3:])

        storage_client = get_storage_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ckpt") as tmp:
            blob.download_to_file(tmp)
            tmp_path = tmp.name

        model = model_class.load_from_checkpoint(tmp_path, map_location=device)
        os.unlink(tmp_path)
        logger.info(f"Model loaded from GCS: {checkpoint_path}")
    else:
        model = model_class.load_from_checkpoint(checkpoint_path, map_location=device)
        logger.info(f"Model loaded from local filesystem: {checkpoint_path}")

    loaded_model_path = checkpoint_path
    model.eval()
    logger.info(f"Model loaded successfully ({current_model_name})")


def get_image_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def _infer_extension(upload: UploadFile) -> str:
    ext = Path(upload.filename or "").suffix.lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        return ext
    ct = (upload.content_type or "").lower()
    if "jpeg" in ct or "jpg" in ct:
        return ".jpg"
    if "png" in ct:
        return ".png"
    return ".png"


def _get_next_daily_index(bucket: storage.Bucket, day_prefix: str) -> int:
    max_idx = 0
    for blob in bucket.list_blobs(prefix=day_prefix):
        name = blob.name.split("/")[-1]
        stem = Path(name).stem
        if stem.isdigit():
            max_idx = max(max_idx, int(stem))
    return max_idx + 1


def _upload_new_object_no_overwrite(bucket: storage.Bucket, object_name: str, data: bytes, content_type: str) -> bool:
    blob = bucket.blob(object_name)
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
        return True
    except Exception:
        return False


def _download_json_array_and_generation(
    bucket: storage.Bucket, object_name: str
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    blob = bucket.blob(object_name)
    try:
        blob.reload()
        generation = blob.generation
        raw = blob.download_as_bytes()
        if not raw:
            return [], generation
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, list):
            return data, generation
        return [data], generation
    except NotFound:
        return [], None


def _upload_json_array_with_generation_match(
    bucket: storage.Bucket, object_name: str, data_list: List[Dict[str, Any]], expected_generation: Optional[int]
) -> None:
    blob = bucket.blob(object_name)
    payload = json.dumps(data_list, ensure_ascii=False, indent=2).encode("utf-8")
    if expected_generation is None:
        blob.upload_from_string(payload, content_type="application/json", if_generation_match=0)
    else:
        blob.upload_from_string(payload, content_type="application/json", if_generation_match=expected_generation)


def _append_to_daily_metadata(bucket: storage.Bucket, daily_metadata_object: str, entry: Dict[str, Any]) -> None:
    for _ in range(20):
        data, gen = _download_json_array_and_generation(bucket, daily_metadata_object)
        data.append(entry)
        try:
            _upload_json_array_with_generation_match(bucket, daily_metadata_object, data, gen)
            return
        except PreconditionFailed:
            continue
    raise RuntimeError("Failed updating daily metadata.json due to concurrent updates.")


def save_request_to_gcs(raw_bytes: bytes, upload: UploadFile, user_label: Optional[str], model_used: str, prediction: dict) -> dict:
    storage_client = get_storage_client()
    bucket = storage_client.bucket(REQUESTS_BUCKET_NAME)

    day_folder = get_folder_date_str()
    day_prefix = f"{REQUESTS_PREFIX}/{day_folder}/"
    ext = _infer_extension(upload)

    idx = _get_next_daily_index(bucket, day_prefix)

    for _ in range(200):
        image_name = f"{idx}{ext}"
        image_object = f"{day_prefix}{image_name}"

        ok = _upload_new_object_no_overwrite(
            bucket=bucket,
            object_name=image_object,
            data=raw_bytes,
            content_type=upload.content_type or "application/octet-stream",
        )
        if ok:
            daily_metadata_object = f"{day_prefix}{DAILY_METADATA_FILENAME}"

            entry = {
                "image_name": image_name,
                "user_label": user_label,  
                "model_output": prediction,
                "model": {"model_name": model_used, "checkpoint_path": loaded_model_path},
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "original_filename": upload.filename,
                "content_type": upload.content_type,
                "bytes": len(raw_bytes),
            }

            _append_to_daily_metadata(bucket, daily_metadata_object, entry)

            return {
                "bucket": REQUESTS_BUCKET_NAME,
                "day_folder": day_folder,
                "index": idx,
                "image_gs_uri": f"gs://{REQUESTS_BUCKET_NAME}/{image_object}",
                "daily_metadata_gs_uri": f"gs://{REQUESTS_BUCKET_NAME}/{daily_metadata_object}",
            }

        idx += 1

    raise RuntimeError("Could not allocate a unique daily image index after many attempts.")


@app.get("/")
async def root():
    load_model_checkpoint()
    return {"status": "ok", "service": "emotion-classifier", "model": current_model_name}


@app.get("/models/")
async def list_models():
    available = get_available_models()
    return {"available_models": available, "current_model": current_model_name, "total": len(available)}


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),

    manual_label: Optional[str] = Form(None),
    model_name: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    accept: Optional[str] = Header(None),
):
   
    if authorization != "dtu":
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail="Invalid authorization header")
    if accept != "application/json":
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Invalid accept header")

   
    manual_label_clean: Optional[str] = None
    if manual_label is not None:
        ml = manual_label.strip().lower()
        if ml == "":
            manual_label_clean = None
        elif ml not in EMOTION_LABELS:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={
                    "error": "Invalid manual label",
                    "message": f"manual_label must be one of: {EMOTION_LABELS}",
                },
            )
        else:
            manual_label_clean = ml

    load_model_checkpoint(model_name=model_name)

    try:
        contents = await file.read()
        if not contents:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"error": "Empty file", "message": "Uploaded file is empty"},
            )

        try:
            image_data = Image.open(io.BytesIO(contents))
        except Exception as e:
            return JSONResponse(
                status_code=HTTPStatus.BAD_REQUEST,
                content={"error": "Invalid image format", "message": f"Could not open image: {str(e)}"},
            )

        if image_data.mode != "L":
            image_data = image_data.convert("L")
        image_data = image_data.resize((48, 48), Image.Resampling.LANCZOS)

        transform = get_image_transform()
        image_tensor = transform(image_data).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        emotion = EMOTION_LABELS[predicted.item()]
        confidence_value = float(confidence.item())
        prob_dict = {EMOTION_LABELS[i]: float(probabilities[0, i].item()) for i in range(len(EMOTION_LABELS))}

        prediction_payload = {"emotion": emotion, "confidence": confidence_value, "probabilities": prob_dict}

        gcs_info = None
        save_error = None

        if SAVE_REQUESTS:
            try:
                gcs_info = save_request_to_gcs(
                    raw_bytes=contents,
                    upload=file,
                    user_label=manual_label_clean,
                    model_used=current_model_name or (model_name or "unknown"),
                    prediction=prediction_payload,
                )
            except (NotFound, Forbidden) as e:
                save_error = f"{type(e).__name__}: {str(e)}"
                logger.error(f"GCS save failed: {save_error}", exc_info=True)
            except Exception as e:
                save_error = f"{type(e).__name__}: {str(e)}"
                logger.error(f"GCS save failed: {save_error}", exc_info=True)

        return {
            "emotion": emotion,
            "confidence": confidence_value,
            "probabilities": prob_dict,
            "manual_label": manual_label_clean,
            "model_used": current_model_name,
            "checkpoint_path": loaded_model_path,
            "saved": bool(gcs_info),
            "gcs": gcs_info,
            "save_error": save_error,
            "message": "Prediction successful",
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Internal server error", "message": str(e)})
