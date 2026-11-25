"""Pydantic schemas: what the API accepts and what it returns."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MAX_TEXT_LENGTH = 5000

Source = Literal["texte", "image", "audio"]
Sentiment = Literal["positif", "neutre", "negatif"]


class TextRequest(BaseModel):
    """Body of POST /predict/text."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"texte": "Livraison rapide et produit parfait, je recommande !"}]},
    )

    # Spaces are stripped first, so "   " is rejected (422).
    texte: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)
    ]


class PredictionResponse(BaseModel):
    """The same answer for every route (text, image, audio, auto)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source": "image",
                    "texte_extrait": "Colis ouvert et article manquant. Service client injoignable !",
                    "sentiment_predit": "negatif",
                    "confiance": 0.89,
                }
            ]
        }
    )

    source: Source
    texte_extrait: str
    sentiment_predit: Sentiment
    confiance: float = Field(ge=0, le=1, description="Probabilité de la classe prédite (0 à 1).")


class ErrorResponse(BaseModel):
    detail: str
