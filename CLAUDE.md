# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

There is no code yet. The only source of truth is the assignment brief `Mini Projet _ ML Appliqué _ 2026.pdf` (in French). Read it before making design decisions. It is an individual, 2-week school project, so keep the code simple and easy to explain in a demo.

The user is learning English: write explanations in simple words and short sentences.

## What to build

A multi-source sentiment classifier for e-commerce customer reviews. Labels are exactly `positif`, `négatif`, `neutre` (French, with the accent on `négatif`).

**Core idea: a cascade architecture.** Every input is turned into plain text first, then one single text classifier makes the decision:

- Text → classifier
- Image → OCR (`pytesseract` + `Pillow`) → classifier
- Audio `.wav` → ASR transcription (`openai-whisper` or a lighter option) → classifier

Keep the OCR and ASR modules separate from the classifier. They only produce text. All routes must share the same prediction function.

## Required pieces (from the brief)

1. **Training script**: load `textes.csv`, minimal cleaning (lowercase, stop-words), `TfidfVectorizer` + Logistic Regression or SVM (scikit-learn). Print a confusion matrix and a classification report. Save `model.pkl` and `vectorizer.pkl`.
2. **Preprocessing modules**: image function (input from `images/`) and audio function (input from `audios/`).
3. **FastAPI app in `main.py`**, with these routes:
   - `POST /predict/text`: JSON body with the text
   - `POST /predict/image`: multipart/form-data image
   - `POST /predict/audio`: multipart/form-data audio
   - `POST /predict`: detects the input type automatically
   - Use Pydantic to validate format, MIME type, and file size.
4. **Test UI**: FastAPI `/docs` (option A) or a Streamlit / static HTML+JS page (option B). Option B must show the predicted sentiment with a confidence indicator.
5. **Tests**: `test_api.py`, runnable with `pytest`, using `TestClient`. Check 200 + JSON shape on every endpoint, and 422 on bad payloads.

Every prediction route returns the same JSON shape:

```json
{
  "source": "...",
  "texte_extrait": "...",
  "sentiment_predit": "...",
  "confiance": 0.89
}
```

## Commands

These come from the brief. They work once the code exists:

- Run the API: `uvicorn main:app --reload`
- Run all tests: `pytest`
- Run one test: `pytest test_api.py::test_name`
- Install deps: `pip install -r requirements.txt` (or `pyproject.toml`)

System dependencies: Tesseract OCR must be installed on Windows and on PATH for `pytesseract`. `openai-whisper` needs `ffmpeg`.

## Deliverables

Source code (training, API, UI), the dataset (`data_test_v2/` or similar), the `.pkl` models, `test_api.py`, and a `README.md` with install steps, run commands for the API and the UI, and screenshots of the UI and Swagger.
