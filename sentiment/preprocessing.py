"""Text cleaning shared by training and by the API.

The same function must run at training time and at prediction time,
so the vectorizer calls it directly (it is saved inside vectorizer.pkl).
"""

import re
import unicodedata

# Small French stop-word list.
# We do NOT remove negation words ("ne", "pas", "non", "rien", "aucun",
# "sans", "jamais") or intensity words ("très", "plus"): they change
# the sentiment of a sentence.
FRENCH_STOP_WORDS = frozenset(
    """
    le la les l un une des du de d au aux et ou a en dans sur pour par avec
    ce cet cette ces c ca je j tu il elle on nous vous ils elles me m te t
    se s lui leur leurs mon ma mes ton ta tes son sa ses notre nos votre vos
    qui que qu quoi dont est sont ete etre ai as avons avez ont avait y
    comme lors
    """.split()
)


def strip_accents(text: str) -> str:
    """'Très déçu' -> 'Tres decu'. OCR and speech-to-text often lose accents."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Lowercase, remove accents, digits, punctuation and stop words."""
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(w for w in text.split() if w not in FRENCH_STOP_WORDS)
