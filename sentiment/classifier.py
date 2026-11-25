"""Load the trained text model (Step 1) and predict a sentiment."""

from functools import lru_cache
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class EmptyTextError(ValueError):
    """Raised when there is no text to classify (empty input, blank image, silent audio)."""


@lru_cache(maxsize=1)
def load_artifacts():
    """Load vectorizer.pkl and model.pkl once, then keep them in memory."""
    vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
    model = joblib.load(MODELS_DIR / "model.pkl")
    return vectorizer, model


def predict_sentiment(text: str) -> tuple[str, float]:
    """Return (label, confidence). Label is 'positif', 'neutre' or 'negatif'."""
    if not text or not text.strip():
        raise EmptyTextError("Aucun texte à analyser.")
    vectorizer, model = load_artifacts()
    probabilities = model.predict_proba(vectorizer.transform([text]))[0]
    best = probabilities.argmax()
    return str(model.classes_[best]), round(float(probabilities[best]), 4)
