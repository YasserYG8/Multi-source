"""Step 4: serve the test web page and the dataset example files.

    GET /                 -> web/index.html
    GET /static/...       -> web/ (CSS, JS, fonts)
    GET /demo/exemples    -> list of example images/audios with their reference sentiment
    GET /demo/fichiers/.. -> the dataset files themselves (only if the dataset folder exists)
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_DIR / "web"
DATASET_DIR = PROJECT_DIR / "dataset_sentiment_ecommerce"

router = APIRouter(tags=["Démo"])


def list_examples(csv_name: str, folder: str, text_column: str) -> list[dict]:
    csv_path = DATASET_DIR / csv_name
    if not csv_path.exists():
        return []
    examples = []
    for _, row in pd.read_csv(csv_path).iterrows():
        # audios.csv says "A01.wav" but the real file is "A01.mp3": match by name stem.
        matches = sorted((DATASET_DIR / folder).glob(Path(row["fichier"]).stem + ".*"))
        if matches:
            examples.append({
                "fichier": matches[0].name,
                "url": f"/demo/fichiers/{folder}/{matches[0].name}",
                "sentiment_ref": row["sentiment_ref"],
                "texte_ref": row[text_column],
            })
    return examples


@router.get("/demo/exemples", summary="Fichiers d'exemple du dataset (pour la page de test)")
def examples():
    return {
        "images": list_examples("images.csv", "images", "texte_attendu_ocr"),
        "audios": list_examples("audios.csv", "audios", "transcription_reference"),
    }


@router.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


def mount_demo(app: FastAPI) -> None:
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    if DATASET_DIR.exists():
        app.mount("/demo/fichiers", StaticFiles(directory=DATASET_DIR), name="dataset")
