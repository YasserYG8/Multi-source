"""Image -> text with Tesseract OCR (pytesseract + Pillow)."""

import io
import os
import shutil
from pathlib import Path
from typing import Union

import pytesseract
from PIL import Image, ImageOps

PROJECT_DIR = Path(__file__).resolve().parent.parent
TESSDATA_DIR = PROJECT_DIR / "tessdata"  # holds fra.traineddata
WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Tesseract is not on PATH on this machine, so find it ourselves.
# The TESSERACT_CMD environment variable can override the path.
pytesseract.pytesseract.tesseract_cmd = (
    os.getenv("TESSERACT_CMD")
    or shutil.which("tesseract")
    or str(WINDOWS_DEFAULT)
)
# Tell Tesseract where fra.traineddata is. (An env variable is safer than
# --tessdata-dir here: the project path contains a space.)
os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)

ImageSource = Union[str, Path, bytes, Image.Image]


def open_image(source: ImageSource) -> Image.Image:
    """Accept a file path, raw bytes (from an upload) or a PIL image."""
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    return Image.open(source)


def preprocess_image(image: Image.Image) -> Image.Image:
    """Grayscale + 2x upscale + auto-contrast: Tesseract reads small text better."""
    image = ImageOps.exif_transpose(image).convert("L")
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    return ImageOps.autocontrast(image)


def extract_text_from_image(source: ImageSource) -> str:
    """Run French OCR and return the text on one line."""
    image = preprocess_image(open_image(source))
    raw_text = pytesseract.image_to_string(image, lang="fra", config="--psm 3")
    return " ".join(raw_text.split())
