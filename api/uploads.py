"""Validation of uploaded files (format, MIME type, size) with Pydantic.

We do not trust the file name or the Content-Type sent by the client:
the real type is detected from the first bytes of the file ("magic bytes").
"""

import io
from typing import Literal, Optional

from fastapi import UploadFile
from fastapi.exceptions import RequestValidationError
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

FileKind = Literal["image", "audio"]

MAX_SIZE = {"image": 5 * 1024 * 1024, "audio": 10 * 1024 * 1024}  # 5 MB / 10 MB

ALLOWED_MIME_TYPES = {
    "image": {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/tiff"},
    "audio": {"audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/webm", "audio/mp4"},
}


def detect_mime_type(data: bytes) -> Optional[str]:
    """Find the real file type from its first bytes. None = unknown format."""
    head = data[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "audio/wav"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if head.startswith(b"OggS"):
        return "audio/ogg"
    if head.startswith(b"fLaC"):
        return "audio/flac"
    if head.startswith(b"\x1a\x45\xdf\xa3"):  # WebM (browser recordings)
        return "audio/webm"
    if head[4:8] == b"ftyp":  # .m4a
        return "audio/mp4"
    return None


def kind_of(mime_type: Optional[str]) -> Optional[FileKind]:
    for kind, mime_types in ALLOWED_MIME_TYPES.items():
        if mime_type in mime_types:
            return kind
    return None


def file_error(message: str) -> PydanticCustomError:
    """A validation error with a clean message (no "Value error, " prefix)."""
    return PydanticCustomError("invalid_file", message)


class UploadedFile(BaseModel):
    """A checked upload. Creating it raises ValidationError if something is wrong."""

    filename: str
    expected_kind: Optional[FileKind] = Field(
        description="'image' or 'audio' for the dedicated routes, None for /predict (auto)."
    )
    mime_type: Optional[str] = Field(description="Type detected from the file content.")
    size: int = Field(ge=0, description="Size in bytes.")
    data: bytes = Field(repr=False)

    @model_validator(mode="after")
    def check_file(self) -> "UploadedFile":
        if self.size == 0:
            raise file_error("Fichier vide.")
        kind = kind_of(self.mime_type)
        if kind is None:
            allowed = sorted(ALLOWED_MIME_TYPES["image"] | ALLOWED_MIME_TYPES["audio"])
            raise file_error(f"Format de fichier non supporté. Formats acceptés : {', '.join(allowed)}.")
        if self.expected_kind and kind != self.expected_kind:
            raise file_error(
                f"Fichier {self.expected_kind} attendu, mais le fichier reçu est de type {self.mime_type}."
            )
        if self.size > MAX_SIZE[kind]:
            raise file_error(f"Fichier trop volumineux : maximum {MAX_SIZE[kind] // (1024 * 1024)} Mo pour un fichier {kind}.")
        if kind == "image":
            try:
                Image.open(io.BytesIO(self.data)).verify()
            except Exception as error:
                raise file_error("Image corrompue ou illisible.") from error
        return self

    @property
    def kind(self) -> FileKind:
        return kind_of(self.mime_type)


async def read_upload(upload: UploadFile, expected_kind: Optional[FileKind], field: str = "fichier") -> UploadedFile:
    """Read the upload (never more than the max size + 1 byte) and validate it.

    Errors become a normal FastAPI 422 response, like any other bad payload.
    """
    data = await upload.read(max(MAX_SIZE.values()) + 1)
    try:
        return UploadedFile(
            filename=upload.filename or "",
            expected_kind=expected_kind,
            mime_type=detect_mime_type(data),
            size=len(data),
            data=data,
        )
    except ValidationError as error:
        errors = error.errors(include_url=False, include_context=False, include_input=False)
        for item in errors:
            item["loc"] = ("body", field)
        raise RequestValidationError(errors) from error
