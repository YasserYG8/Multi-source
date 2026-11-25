"""Integration tests: the HTTP API, through FastAPI's TestClient (httpx).

For every route we check:
  - 200 OK with the unified JSON answer for a good request
  - 422 Unprocessable Entity for bad payloads

Run:  pytest              (all tests)
      pytest -m "not slow" (skip the Whisper tests)
"""

import pytest

from api.uploads import MAX_SIZE
from sentiment.preprocessing import strip_accents

from .conftest import make_blank_image, make_silent_wav, make_text_image, needs_voice_file

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SENTIMENTS = {"positif", "neutre", "negatif"}


def assert_prediction(response, source: str) -> dict:
    """Check the unified answer: same keys and types for every route."""
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"source", "texte_extrait", "sentiment_predit", "confiance"}
    assert body["source"] == source
    assert isinstance(body["texte_extrait"], str) and body["texte_extrait"].strip()
    assert body["sentiment_predit"] in SENTIMENTS
    assert isinstance(body["confiance"], float)
    assert 0.0 <= body["confiance"] <= 1.0
    return body


def assert_422(response, message_part: str = "") -> list:
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail, "422 errors always use the list format"
    assert all("msg" in item and "loc" in item for item in detail)
    if message_part:
        assert any(message_part in item["msg"] for item in detail), detail
    return detail


def normalize(text: str) -> str:
    return strip_accents(text.lower())


# ---------------------------------------------------------------- system

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_lists_the_four_prediction_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/predict/text", "/predict/image", "/predict/audio", "/predict"):
        assert "post" in paths[path]


def test_test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert client.get("/static/app.js").status_code == 200


def test_demo_examples_list(client):
    body = client.get("/demo/exemples").json()
    assert set(body) == {"images", "audios"}
    for example in body["images"] + body["audios"]:  # empty lists when the dataset is missing
        assert example["sentiment_ref"] in SENTIMENTS
        assert client.get(example["url"]).status_code == 200


# ---------------------------------------------------------------- /predict/text

@pytest.mark.parametrize(
    "texte, expected",
    [
        # Reviews from the training set: the model must get these right.
        ("Produit conforme à la description, livraison super rapide en 24h, je recommande vivement !", "positif"),
        ("Article reçu cassé et le service client ne répond pas depuis une semaine. Très déçu.", "negatif"),
        ("Le colis est arrivé dans les délais annoncés. L'emballage est standard.", "neutre"),
    ],
)
def test_text_known_reviews(client, texte, expected):
    body = assert_prediction(client.post("/predict/text", json={"texte": texte}), "texte")
    assert body["sentiment_predit"] == expected
    assert body["texte_extrait"] == texte


def test_text_is_stripped(client):
    body = assert_prediction(client.post("/predict/text", json={"texte": "  Très bon produit  "}), "texte")
    assert body["texte_extrait"] == "Très bon produit"


@pytest.mark.parametrize(
    "payload",
    [
        {},                                # missing field
        {"texte": ""},                     # empty
        {"texte": "    "},                 # only spaces
        {"texte": 42},                     # wrong type
        {"texte": None},                   # null
        {"texte": "ok", "note": 5},        # unknown field
        {"texte": "a" * 5001},             # too long
        ["texte", "ok"],                   # not an object
    ],
    ids=["missing", "empty", "spaces", "number", "null", "extra-field", "too-long", "list"],
)
def test_text_invalid_payload_returns_422(client, payload):
    assert_422(client.post("/predict/text", json=payload))


def test_text_without_body_returns_422(client):
    assert_422(client.post("/predict/text"))


def test_text_with_broken_json_returns_422(client):
    response = client.post("/predict/text", content=b'{"texte": ', headers={"Content-Type": "application/json"})
    assert_422(response)


# ---------------------------------------------------------------- /predict/image

def test_image_ocr_and_prediction(client, review_image):
    response = client.post("/predict/image", files={"fichier": ("avis.png", review_image, "image/png")})
    body = assert_prediction(response, "image")
    assert "recommande" in normalize(body["texte_extrait"])
    assert body["sentiment_predit"] == "positif"


def test_image_jpeg_is_accepted(client):
    jpeg = make_text_image("Livraison rapide, produit parfait.", image_format="JPEG")
    assert_prediction(client.post("/predict/image", files={"fichier": ("avis.jpg", jpeg, "image/jpeg")}), "image")


def test_image_type_comes_from_content_not_from_name(client, review_image):
    # A real PNG with a wrong name and a wrong Content-Type is still accepted.
    response = client.post("/predict/image", files={"fichier": ("avis.txt", review_image, "text/plain")})
    assert_prediction(response, "image")


def test_image_missing_file_returns_422(client):
    assert_422(client.post("/predict/image"))


def test_image_empty_file_returns_422(client):
    assert_422(client.post("/predict/image", files={"fichier": ("vide.png", b"", "image/png")}), "Fichier vide")


def test_image_text_file_returns_422(client):
    response = client.post("/predict/image", files={"fichier": ("faux.png", b"ceci n'est pas une image", "image/png")})
    assert_422(response, "non supporté")


def test_image_route_refuses_audio(client):
    response = client.post("/predict/image", files={"fichier": ("son.wav", make_silent_wav(), "audio/wav")})
    assert_422(response, "image attendu")


def test_image_corrupted_returns_422(client):
    response = client.post("/predict/image", files={"fichier": ("casse.png", PNG_SIGNATURE + b"\x00" * 100, "image/png")})
    assert_422(response, "corrompue")


def test_image_too_big_returns_422(client):
    too_big = PNG_SIGNATURE + b"\x00" * MAX_SIZE["image"]
    assert_422(client.post("/predict/image", files={"fichier": ("gros.png", too_big, "image/png")}), "trop volumineux")


def test_image_without_text_returns_422(client):
    response = client.post("/predict/image", files={"fichier": ("blanc.png", make_blank_image(), "image/png")})
    assert_422(response, "Aucun texte")


# ---------------------------------------------------------------- /predict/audio

@pytest.mark.slow
@needs_voice_file
def test_audio_transcription_and_prediction(client, voice_file):
    response = client.post("/predict/audio", files={"fichier": ("A01.mp3", voice_file, "audio/mpeg")})
    body = assert_prediction(response, "audio")
    assert "ravi" in normalize(body["texte_extrait"])
    assert body["sentiment_predit"] == "positif"


@pytest.mark.slow
def test_audio_silence_returns_422(client):
    # Without the VAD filter, Whisper invents a sentence here.
    response = client.post("/predict/audio", files={"fichier": ("silence.wav", make_silent_wav(), "audio/wav")})
    assert_422(response, "Aucun texte")


def test_audio_missing_file_returns_422(client):
    assert_422(client.post("/predict/audio"))


def test_audio_empty_file_returns_422(client):
    assert_422(client.post("/predict/audio", files={"fichier": ("vide.mp3", b"", "audio/mpeg")}), "Fichier vide")


def test_audio_route_refuses_image(client, review_image):
    response = client.post("/predict/audio", files={"fichier": ("avis.png", review_image, "image/png")})
    assert_422(response, "audio attendu")


def test_audio_corrupted_returns_422(client):
    response = client.post("/predict/audio", files={"fichier": ("casse.mp3", b"ID3" + b"\x00" * 500, "audio/mpeg")})
    assert_422(response, "illisible")


def test_audio_too_big_returns_422(client):
    too_big = b"ID3" + b"\x00" * MAX_SIZE["audio"]
    assert_422(client.post("/predict/audio", files={"fichier": ("gros.mp3", too_big, "audio/mpeg")}), "trop volumineux")


# ---------------------------------------------------------------- /predict (auto)

def test_auto_text(client):
    response = client.post("/predict", data={"texte": "Colis arrivé cassé, très déçu."})
    assert_prediction(response, "texte")


def test_auto_detects_image(client, review_image):
    # No useful name or type: the API must find "image" from the bytes.
    response = client.post("/predict", files={"fichier": ("fichier", review_image, "application/octet-stream")})
    assert_prediction(response, "image")


@pytest.mark.slow
@needs_voice_file
def test_auto_detects_audio(client, voice_file):
    response = client.post("/predict", files={"fichier": ("fichier", voice_file, "application/octet-stream")})
    assert_prediction(response, "audio")


def test_auto_without_input_returns_422(client):
    assert_422(client.post("/predict"), "un seul champ")


def test_auto_with_text_and_file_returns_422(client, review_image):
    response = client.post("/predict", data={"texte": "Bien"}, files={"fichier": ("avis.png", review_image, "image/png")})
    assert_422(response, "un seul champ")


def test_auto_with_only_spaces_returns_422(client):
    assert_422(client.post("/predict", data={"texte": "   "}), "un seul champ")


def test_auto_unsupported_file_returns_422(client):
    response = client.post("/predict", files={"fichier": ("doc.pdf", b"%PDF-1.7 ...", "application/pdf")})
    assert_422(response, "non supporté")


# ---------------------------------------------------------------- same answer everywhere

@pytest.mark.slow
@needs_voice_file
def test_all_routes_return_the_same_structure(client, review_image, voice_file):
    responses = [
        client.post("/predict/text", json={"texte": "Très satisfait"}),
        client.post("/predict/image", files={"fichier": ("a.png", review_image, "image/png")}),
        client.post("/predict/audio", files={"fichier": ("a.mp3", voice_file, "audio/mpeg")}),
        client.post("/predict", data={"texte": "Très satisfait"}),
    ]
    key_sets = {frozenset(response.json()) for response in responses}
    assert all(response.status_code == 200 for response in responses)
    assert len(key_sets) == 1
