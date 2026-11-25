"""Shared fixtures. Test files are generated in code, so the tests do not
need the dataset folder (it is not in git). Only the real-voice audio tests
use the dataset, and they are skipped when it is missing."""

import io
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from main import app

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset_sentiment_ecommerce"
VOICE_FILE = DATASET_DIR / "audios" / "A01.mp3"  # "Bonjour, je suis vraiment ravi de ma commande..." (positif)

needs_voice_file = pytest.mark.skipif(not VOICE_FILE.exists(), reason="dataset audio file not available")


@pytest.fixture(scope="session")
def client():
    # "with" runs the app's startup code (model loading), like uvicorn does.
    with TestClient(app) as test_client:
        yield test_client


def make_text_image(*lines: str, image_format: str = "PNG") -> bytes:
    """A white image with black text, like a screenshot of a review."""
    font = ImageFont.load_default(size=36)
    image = Image.new("RGB", (1000, 80 + 70 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((30, 40 + 70 * index), line, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def make_blank_image() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 200), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def make_silent_wav(seconds: float = 2.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buffer.getvalue()


@pytest.fixture(scope="session")
def review_image() -> bytes:
    return make_text_image("Produit excellent, livraison rapide.", "Je recommande vivement !")


@pytest.fixture(scope="session")
def voice_file() -> bytes:
    return VOICE_FILE.read_bytes()
