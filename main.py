"""FastAPI app: multi-source sentiment classifier (text, image, audio).

Run:  uvicorn main:app --reload
Page: http://127.0.0.1:8000/       (test web page)
Docs: http://127.0.0.1:8000/docs   (Swagger)
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.demo import mount_demo
from api.routes import router
from sentiment.asr import InvalidAudioError, get_whisper_model
from sentiment.classifier import EmptyTextError, load_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()  # load the .pkl files at startup, not on the first request
    # Whisper takes ~15 s to load: do it in the background so the server starts
    # at once and the first audio request (e.g. during a demo) is not slow.
    threading.Thread(target=get_whisper_model, daemon=True).start()
    yield


app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    description=(
        "Prédit le sentiment (positif, neutre, negatif) d'un avis client e-commerce, "
        "qu'il soit écrit, en image ou en audio.\n\n"
        "**Architecture en cascade** : image → OCR (Tesseract) → texte, "
        "audio → transcription (Whisper) → texte, puis un seul modèle texte "
        "(TF-IDF + Régression Logistique) prend la décision."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# The test web page (Step 4) may be opened from another origin (file:// or another port).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(router)
mount_demo(app)  # test web page at "/" (Step 4)


@app.exception_handler(EmptyTextError)
@app.exception_handler(InvalidAudioError)
async def unprocessable_input(request: Request, error: ValueError):
    """No text found in the image/audio, or unreadable audio -> 422.

    Same shape as FastAPI's own 422 errors, so clients read all errors the same way.
    """
    content = {"detail": [{"type": "unprocessable_input", "loc": ["body"], "msg": str(error)}]}
    return JSONResponse(status_code=422, content=content)


@app.get("/health", tags=["Système"], summary="Vérifier que l'API fonctionne")
def health():
    return {"status": "ok"}
