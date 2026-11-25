# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS/JS (no framework, no build step), served by the FastAPI app itself at `/`. Chosen by the user over Streamlit.

## Users

A teacher or jury watching a live demo of an individual school project (Applied ML mini-project, 2026), usually on a projector. The student drives the page; the audience reads it from a distance.

## Product Purpose

A test page for a multi-source sentiment classifier for e-commerce customer reviews. The visitor sends a review as text, image (screenshot / scanned ticket) or audio (voice message), and sees the predicted sentiment (`positif`, `neutre`, `negatif`) with a confidence score. Success: the demo shows clearly that all three sources work and end in the same answer.

## Positioning

Cascade architecture: image goes through OCR (Tesseract) and audio through transcription (faster-whisper); both become plain text, and one single text model (TF-IDF + Logistic Regression) makes the decision. The page should make this "everything becomes text first" path visible.

## Operating Context

Runs locally: `uvicorn main:app --reload`, then http://127.0.0.1:8000/. API routes: `POST /predict/text` (JSON `{"texte"}`), `/predict/image`, `/predict/audio` (multipart field `fichier`), `/predict` (auto). Every route returns `{source, texte_extrait, sentiment_predit, confiance}`. Errors are 422 with `{"detail": [{"msg": ...}]}`. OCR takes ~1 s, audio transcription several seconds on CPU.

## Capabilities and Constraints

- UI language: French.
- Sentiment labels come from the API without accent (`negatif`); display text may use `négatif`.
- Confidence values are low by nature (~0.4–0.8) because the model is trained on 24 reviews; the page must not present them as alarming or as certainty.
- Limits: images 5 MB (PNG, JPEG, WEBP, BMP, TIFF), audio 10 MB (WAV, MP3, OGG, FLAC, WEBM, M4A).
- Demo files exist in `dataset_sentiment_ecommerce/images/` and `audios/`.

## Evidence on Hand

Real dataset files and measured results: 89% accuracy on 18 held-out texts; 8/9 correct for images and for audio (`reports/`). No other claims.

## Product Principles

1. Show the path, not just the verdict: extracted text is always visible next to the result.
2. Honest confidence: show the number as it is, with context.
3. Demo-proof: big readable result, clear errors, works offline on localhost.
