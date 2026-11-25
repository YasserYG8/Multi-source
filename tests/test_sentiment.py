"""Unit tests: the building blocks, without HTTP."""

import pytest

from api.uploads import detect_mime_type, kind_of
from sentiment.classifier import EmptyTextError, load_artifacts, predict_sentiment
from sentiment.ocr import extract_text_from_image
from sentiment.pipeline import predict_text
from sentiment.preprocessing import clean_text

from .conftest import DATASET_DIR, make_blank_image, make_silent_wav, make_text_image

needs_dataset_image = pytest.mark.skipif(
    not (DATASET_DIR / "images" / "I02.png").exists(), reason="dataset image not available"
)


# ---------------------------------------------------------------- preprocessing

def test_clean_text_lowercases_and_removes_accents():
    assert clean_text("TRÈS Déçu") == "tres decu"


def test_clean_text_removes_punctuation_digits_and_stop_words():
    assert clean_text("Le colis est arrivé en 24h !") == "colis arrive h"


def test_clean_text_keeps_negation_words():
    # "ne ... pas" changes the sentiment, so it must stay.
    assert clean_text("Il ne fonctionne pas, aucun intérêt, sans plus.") == "ne fonctionne pas aucun interet sans plus"


# ---------------------------------------------------------------- model

def test_model_knows_the_three_classes():
    _vectorizer, model = load_artifacts()
    assert set(model.classes_) == {"positif", "neutre", "negatif"}


def test_predict_sentiment_returns_label_and_probability():
    label, confidence = predict_sentiment("Livraison rapide, produit parfait.")
    assert label in {"positif", "neutre", "negatif"}
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_predict_sentiment_refuses_empty_text(text):
    with pytest.raises(EmptyTextError):
        predict_sentiment(text)


def test_pipeline_returns_the_unified_answer():
    result = predict_text("Très satisfait de mon achat")
    assert set(result) == {"source", "texte_extrait", "sentiment_predit", "confiance"}
    assert result["source"] == "texte"


# ---------------------------------------------------------------- OCR

def test_ocr_reads_generated_image():
    # Pillow's built-in font has no accented letters, so this text has none.
    image = make_text_image("Colis ouvert et article manquant.")
    assert extract_text_from_image(image) == "Colis ouvert et article manquant."


@needs_dataset_image
def test_ocr_reads_french_accents():
    # I02.png: "RÉCLAMATION COMMANDE #9921 ... injoignable, inadmissible !"
    text = extract_text_from_image(DATASET_DIR / "images" / "I02.png")
    assert "RÉCLAMATION" in text  # accents need the French language file (tessdata/fra)
    assert "inadmissible" in text


def test_ocr_on_blank_image_returns_empty_text():
    assert extract_text_from_image(make_blank_image()) == ""


# ---------------------------------------------------------------- file type detection

@pytest.mark.parametrize(
    "data, mime_type",
    [
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 8, "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"BM" + b"\x00" * 14, "image/bmp"),
        (b"II*\x00" + b"\x00" * 12, "image/tiff"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav"),
        (b"ID3\x04" + b"\x00" * 12, "audio/mpeg"),
        (b"\xff\xfb\x90\x00" + b"\x00" * 12, "audio/mpeg"),
        (b"OggS" + b"\x00" * 12, "audio/ogg"),
        (b"fLaC" + b"\x00" * 12, "audio/flac"),
        (b"\x1a\x45\xdf\xa3" + b"\x00" * 12, "audio/webm"),
        (b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 4, "audio/mp4"),
        (b"%PDF-1.7", None),
        (b"bonjour", None),
        (b"", None),
    ],
)
def test_detect_mime_type(data, mime_type):
    assert detect_mime_type(data) == mime_type


def test_detect_real_generated_files():
    assert detect_mime_type(make_text_image("x")) == "image/png"
    assert detect_mime_type(make_text_image("x", image_format="JPEG")) == "image/jpeg"
    assert detect_mime_type(make_silent_wav(0.1)) == "audio/wav"


def test_kind_of():
    assert kind_of("image/png") == "image"
    assert kind_of("audio/mpeg") == "audio"
    assert kind_of("application/pdf") is None
    assert kind_of(None) is None
