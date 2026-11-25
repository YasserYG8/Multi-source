"""The cascade: every input becomes text, then the same text model decides.

    texte ------------------------------> predict_sentiment
    image --> extract_text_from_image --> predict_sentiment
    audio --> transcribe_audio ---------> predict_sentiment

All functions return the same dict (the unified JSON answer of the API).
"""

from .asr import AudioSource, transcribe_audio
from .classifier import predict_sentiment
from .ocr import ImageSource, extract_text_from_image


def build_result(source: str, text: str) -> dict:
    sentiment, confidence = predict_sentiment(text)
    return {
        "source": source,
        "texte_extrait": text,
        "sentiment_predit": sentiment,
        "confiance": confidence,
    }


def predict_text(text: str) -> dict:
    return build_result("texte", text.strip())


def predict_image(image: ImageSource) -> dict:
    return build_result("image", extract_text_from_image(image))


def predict_audio(audio: AudioSource) -> dict:
    return build_result("audio", transcribe_audio(audio))
