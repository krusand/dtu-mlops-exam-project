"""Unit tests for the emotion classification API."""

import io
import os
import pytest
import requests
import time
from pathlib import Path
from http import HTTPStatus
from PIL import Image

# Use cloud API or local API
API_URL = os.getenv(
    "API_URL",
    "https://emotion-classifier-597500488480.europe-west1.run.app"
)

# Request settings for cloud resilience
REQUEST_TIMEOUT = 120  # 2 minutes for slow cold starts
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def _request_with_retry(method, url, max_retries=MAX_RETRIES, **kwargs):
    """Make HTTP request with exponential backoff retry logic."""
    kwargs.setdefault('timeout', REQUEST_TIMEOUT)
    
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            # Retry on 429 (rate limit) or 5xx errors
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"\n[Retry {attempt + 1}/{max_retries}] Got 429, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            return response
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"\n[Retry {attempt + 1}/{max_retries}] Timeout, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            raise
    
    return response


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Try to find an actual image first
    data_dir = Path("data/raw/datasets/msambare/fer2013/versions/1/train")
    if data_dir.exists():
        for emotion_dir in sorted(data_dir.iterdir()):
            if emotion_dir.is_dir():
                for img in emotion_dir.glob('*.jpg'):
                    return img
    
    # If no image found, create a test image in memory
    img = Image.new('L', (48, 48), color=128)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def _get_file_obj(sample_image):
    """Helper to get file object from either Path or BytesIO."""
    if isinstance(sample_image, Path):
        return open(sample_image, 'rb'), True
    else:
        sample_image.seek(0)
        return sample_image, False


class TestHeaderValidation:
    """Test API header validation."""

    def test_missing_authorization_header(self, sample_image):
        """Test that request without authorization header is rejected."""
        f, should_close = _get_file_obj(sample_image)
        try:
            response = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f},
                headers={"accept": "application/json"}
            )
            assert response.status_code in [
                HTTPStatus.UNAUTHORIZED,
                HTTPStatus.BAD_REQUEST
            ]
        finally:
            if should_close:
                f.close()

    def test_invalid_authorization_header(self, sample_image):
        """Test that request with invalid authorization is rejected."""
        f, should_close = _get_file_obj(sample_image)
        try:
            response = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "invalid-key",
                    "accept": "application/json"
                }
            )
            assert response.status_code == HTTPStatus.UNAUTHORIZED
        finally:
            if should_close:
                f.close()

    def test_missing_accept_header(self, sample_image):
        """Test that request without accept header is rejected."""
        f, should_close = _get_file_obj(sample_image)
        try:
            response = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f},
                headers={"authorization": "dtu"}
            )
            # Should get 400 error for missing accept header
            assert response.status_code == HTTPStatus.BAD_REQUEST
        finally:
            if should_close:
                f.close()

    def test_invalid_accept_header(self, sample_image):
        """Test that request with invalid accept header is rejected."""
        f, should_close = _get_file_obj(sample_image)
        try:
            response = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "text/html"
                }
            )
            # Should get 400 error for invalid accept header
            assert response.status_code == HTTPStatus.BAD_REQUEST
        finally:
            if should_close:
                f.close()


class TestSuccessfulPrediction:
    """Test successful API predictions."""

    def test_predict_success_returns_valid_response(self, sample_image):
        """Test successful prediction with correct headers."""
        f, should_close = _get_file_obj(sample_image)
        try:
            response = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )
            assert response.status_code == HTTPStatus.OK
            result = response.json()

            # Check all required fields are present
            required_fields = {
                "emotion", "confidence", "probabilities", "message"
            }
            assert required_fields.issubset(result.keys())

            # Check emotion is valid
            valid_emotions = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}
            assert result["emotion"] in valid_emotions

            # Check confidence is between 0 and 1
            assert 0 <= result["confidence"] <= 1
        finally:
            if should_close:
                f.close()

    def test_predict_response_is_json(self, sample_image):
        """Test that response content-type is JSON."""
        f, should_close = _get_file_obj(sample_image)
        try:
            response = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )
            assert response.status_code == HTTPStatus.OK
            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type
        finally:
            if should_close:
                f.close()


class TestFileHandling:
    """Test API file handling."""

    def test_predict_no_file(self):
        """Test that request without file is rejected."""
        response = _request_with_retry(
            "POST",
            f"{API_URL}/predict/",
            headers={
                "authorization": "dtu",
                "accept": "application/json"
            }
        )

        # Should return error when no file is provided
        assert response.status_code in [
            HTTPStatus.BAD_REQUEST,
            HTTPStatus.UNPROCESSABLE_ENTITY
        ]

    def test_predict_with_valid_file(self, sample_image):
        """Test that valid file is accepted."""
        f, should_close = _get_file_obj(sample_image)
        try:
            response = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )
            assert response.status_code == HTTPStatus.OK
        finally:
            if should_close:
                f.close()


class TestConsistency:
    """Test API consistency and reproducibility."""

    def test_same_image_produces_same_prediction(self, sample_image):
        """Test that the same image produces the same prediction."""
        # First prediction
        f1, should_close1 = _get_file_obj(sample_image)
        try:
            response1 = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f1},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )
            result1 = response1.json()
        finally:
            if should_close1:
                f1.close()

        # Second prediction with same image
        f2, should_close2 = _get_file_obj(sample_image)
        try:
            response2 = _request_with_retry(
                "POST",
                f"{API_URL}/predict/",
                files={"file": f2},
                headers={
                    "authorization": "dtu",
                    "accept": "application/json"
                }
            )
            result2 = response2.json()
        finally:
            if should_close2:
                f2.close()

        # Both should produce identical results
        assert result1["emotion"] == result2["emotion"]
        assert result1["confidence"] == result2["confidence"]
        assert result1["probabilities"] == result2["probabilities"]
