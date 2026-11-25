"""Step 1: train the text sentiment classifier.

Train set : textes.csv (24 reviews).
Test set  : the reference texts of images.csv and audios.csv (18 reviews).
            The model never sees them during training.

Outputs:
    models/vectorizer.pkl, models/model.pkl
    reports/classification_report.txt, reports/confusion_matrix.png

Run:  python train.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save images without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from sentiment.preprocessing import clean_text

DATA_DIR = Path("dataset_sentiment_ecommerce")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
LABELS = ["positif", "neutre", "negatif"]


def load_data():
    train = pd.read_csv(DATA_DIR / "textes.csv")
    images = pd.read_csv(DATA_DIR / "images.csv")
    audios = pd.read_csv(DATA_DIR / "audios.csv")

    x_train, y_train = train["texte"], train["sentiment_ref"]
    x_test = pd.concat([images["texte_attendu_ocr"], audios["transcription_reference"]])
    y_test = pd.concat([images["sentiment_ref"], audios["sentiment_ref"]])
    return x_train, y_train, x_test, y_test


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # print accents correctly on Windows
    x_train, y_train, x_test, y_test = load_data()
    print(f"Train: {len(x_train)} texts | Test: {len(x_test)} texts")

    # Character n-grams inside word boundaries: robust to OCR typos and to
    # French word forms (livré / livraison) on a very small dataset.
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb",
        ngram_range=(2, 5),
        sublinear_tf=True,
    )
    model = LogisticRegression(C=10, max_iter=1000)

    model.fit(vectorizer.fit_transform(x_train), y_train)

    y_pred = model.predict(vectorizer.transform(x_test))
    report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0)
    print(report)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.pkl")
    joblib.dump(model, MODELS_DIR / "model.pkl")
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=LABELS, cmap="Blues")
    plt.title("Confusion matrix (test: image + audio texts)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)

    print(f"Saved models to {MODELS_DIR}/ and reports to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
