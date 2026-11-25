"""Audio -> text with faster-whisper (a light, fast version of OpenAI Whisper).

Works with .wav and .mp3 (faster-whisper decodes audio itself).
"""

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Union

from faster_whisper import WhisperModel

# "small" is a good balance for French on CPU. Change it with the
# WHISPER_MODEL environment variable ("tiny", "base", "medium"...).
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")

AudioSource = Union[str, Path, bytes]


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    """Load the model once (it takes a few seconds), then reuse it."""
    return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def transcribe_audio(source: AudioSource) -> str:
    """Return the French transcription of an audio file or raw audio bytes."""
    audio = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    segments, _info = get_whisper_model().transcribe(audio, language="fr", beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()
