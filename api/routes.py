"""Prediction routes. Every route returns the same PredictionResponse."""

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError

from sentiment.pipeline import predict_audio, predict_image, predict_text

from .schemas import MAX_TEXT_LENGTH, ErrorResponse, PredictionResponse, TextRequest
from .uploads import UploadedFile, read_upload

router = APIRouter(
    prefix="/predict",
    tags=["Prédiction"],
    responses={422: {"description": "Données invalides (format, type MIME, taille, texte vide…)"}},
)


async def run_pipeline(upload: UploadedFile) -> dict:
    """OCR and Whisper are slow CPU work: run them in a thread so the server stays responsive."""
    pipeline = predict_image if upload.kind == "image" else predict_audio
    return await run_in_threadpool(pipeline, upload.data)


@router.post("/text", response_model=PredictionResponse, summary="Avis texte")
def predict_from_text(request: TextRequest):
    """Classe directement un avis écrit."""
    return predict_text(request.texte)


@router.post("/image", response_model=PredictionResponse, summary="Avis image (OCR)")
async def predict_from_image(
    fichier: UploadFile = File(description="PNG, JPEG, WEBP, BMP ou TIFF. 5 Mo maximum."),
):
    """Extrait le texte de l'image avec Tesseract (OCR), puis le classe."""
    return await run_pipeline(await read_upload(fichier, expected_kind="image"))


@router.post("/audio", response_model=PredictionResponse, summary="Avis audio (transcription)")
async def predict_from_audio(
    fichier: UploadFile = File(description="WAV, MP3, OGG, FLAC, WEBM ou M4A. 10 Mo maximum."),
):
    """Transcrit l'audio avec Whisper, puis classe le texte obtenu."""
    return await run_pipeline(await read_upload(fichier, expected_kind="audio"))


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Détection automatique du type",
    responses={422: {"model": ErrorResponse}},
)
async def predict_auto(
    texte: Optional[str] = Form(None, max_length=MAX_TEXT_LENGTH, description="Un avis écrit…"),
    fichier: Optional[UploadFile] = File(None, description="…ou un fichier image / audio."),
):
    """Envoyez **soit** `texte`, **soit** `fichier`. Le type du fichier est détecté
    à partir de son contenu (pas de son nom), puis le bon pipeline est utilisé."""
    has_text = texte is not None and texte.strip() != ""
    has_file = fichier is not None and fichier.filename
    if has_text == bool(has_file):
        raise RequestValidationError([{
            "type": "missing" if not has_text else "invalid_input",
            "loc": ("body",),
            "msg": "Envoyez exactement un seul champ : 'texte' ou 'fichier'.",
        }])
    if has_text:
        return predict_text(texte)
    return await run_pipeline(await read_upload(fichier, expected_kind=None))
