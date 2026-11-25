"""Step 2 check: run the full cascade on every image and audio file.

For each file we compare:
  - the extracted text with the reference text (similarity 0..1)
  - the predicted sentiment with the reference sentiment

Output: reports/pipelines_results.csv
Run:    python evaluate_pipelines.py
"""

import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from sentiment.pipeline import predict_audio, predict_image
from sentiment.preprocessing import clean_text

DATA_DIR = Path("dataset_sentiment_ecommerce")
REPORTS_DIR = Path("reports")


def similarity(a: str, b: str) -> float:
    """How close two texts are, ignoring case, accents and punctuation."""
    return round(SequenceMatcher(None, clean_text(a), clean_text(b)).ratio(), 2)


def find_file(folder: Path, name: str) -> Path:
    """audios.csv says 'A01.wav' but the real file is 'A01.mp3': match by name stem."""
    matches = list(folder.glob(Path(name).stem + ".*"))
    if not matches:
        raise FileNotFoundError(folder / name)
    return matches[0]


def evaluate(kind, csv_name, folder, text_column, predict):
    rows = []
    for _, row in pd.read_csv(DATA_DIR / csv_name).iterrows():
        path = find_file(DATA_DIR / folder, row["fichier"])
        result = predict(path)
        rows.append({
            "type": kind,
            "fichier": path.name,
            "texte_extrait": result["texte_extrait"],
            "similarite_texte": similarity(result["texte_extrait"], row[text_column]),
            "sentiment_ref": row["sentiment_ref"],
            "sentiment_predit": result["sentiment_predit"],
            "confiance": result["confiance"],
            "correct": result["sentiment_predit"] == row["sentiment_ref"],
        })
        mark = "OK " if rows[-1]["correct"] else "BAD"
        print(f"[{mark}] {path.name}  ref={row['sentiment_ref']:8s} pred={result['sentiment_predit']:8s} "
              f"conf={result['confiance']:.2f}  sim={rows[-1]['similarite_texte']:.2f}")
        print(f"       {result['texte_extrait']}")
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = evaluate("image", "images.csv", "images", "texte_attendu_ocr", predict_image)
    rows += evaluate("audio", "audios.csv", "audios", "transcription_reference", predict_audio)

    results = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORTS_DIR / "pipelines_results.csv", index=False, encoding="utf-8")

    print("\nSummary per input type:")
    print(results.groupby("type")[["correct", "similarite_texte", "confiance"]].mean().round(2))


if __name__ == "__main__":
    main()
