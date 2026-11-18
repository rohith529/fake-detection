"""Stage 2 — NLP preprocessing: normalise → contraction expansion → tokenize → negation-aware stop words → stem.

Remediated architecture:
1. Eradicated synthetic __QLEAD__ and __DLEAD__ bias tokens.
   Debunking and fact-checking headlines are evaluated objectively on their merit,
   not artificially penalized because of their headline structure.
2. Contraction expansion: Expands n't, won't, can't so negation tokens are preserved.
3. Negation-aware filtering: Retains negation terms (not, no, never, neither, nor, etc.)
   rather than purging them via generic stopword lists, preserving semantic polarity.
4. Mojibake & Encoding Repair: Detects and fixes Latin-1 / Windows-1252 to UTF-8 corruption.
"""
import re
import unicodedata
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()
_URL = re.compile(r"https?://\S+|www\.\S+")
_AGENCY = re.compile(r"^.{0,80}?\((reuters|ap|afp)\)\s*[-–—]\s*", re.I)
_JUNK = re.compile(
    r"\(?\b(reuters|associated press|afp)\b\)?"
    r"|featured image( via| by)?|image via|getty images|screenshot"
    r"|pic\.twitter\.com/\S+|@\w+"
    r"|21st century wire|read more:?|via \w+", re.I
)

# Negation terms that must NEVER be stripped as stopwords
NEGATION_WORDS = frozenset({
    "not", "no", "never", "neither", "nor", "none",
    "without", "barely", "hardly", "scarcely"
})

# Custom stop words preserving polarity
SAFE_STOP_WORDS = frozenset(w for w in ENGLISH_STOP_WORDS if w not in NEGATION_WORDS)

# Common contractions mapped to expanded forms
_CONTRACTIONS = {
    r"\bcan'?t\b": "can not",
    r"\bwon'?t\b": "will not",
    r"\bshan'?t\b": "shall not",
    r"\bn'?t\b": " not",
    r"\b've\b": " have",
    r"\b're\b": " are",
    r"\b'd\b": " would",
    r"\b'll\b": " will",
    r"\b'm\b": " am"
}
_CONTRACTION_RES = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in _CONTRACTIONS.items()]


def fix_encoding(text: str) -> str:
    """Repair mojibake (e.g. Windows-1252 / Latin-1 read as UTF-8) and normalize unicode."""
    if not isinstance(text, str):
        return ""
    try:
        text.encode("utf-8").decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(enc).decode("utf-8")
            orig_bad = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
            new_bad = sum(1 for c in repaired if ord(c) > 127) / max(len(repaired), 1)
            if new_bad < orig_bad:
                text = repaired
                break
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
    return unicodedata.normalize("NFKC", text)


def expand_contractions(text: str) -> str:
    """Expand contractions to preserve explicit negation and auxiliary verbs."""
    for pattern, repl in _CONTRACTION_RES:
        text = pattern.sub(repl, text)
    return text


def strip_leakage(text: str) -> str:
    """Remove dataset-collection artefacts before processing."""
    text = _AGENCY.sub("", text)
    return _JUNK.sub(" ", text)


def tokenize(text: str) -> list[str]:
    """Tokenize lowercase text after stripping URLs and expanding contractions."""
    text = expand_contractions(text.lower())
    text = _URL.sub(" ", text)
    return re.findall(r"[a-z]+", text)


def stem(word: str) -> str:
    """Porter stemmer wrapper."""
    return _stemmer.stem(word)


def clean(text: str, preserve_stem: bool = True) -> str:
    """Full preprocessing pipeline.
    
    1. Fixes encoding and normalizes text.
    2. Expands contractions.
    3. Tokenizes and filters non-negation stop words.
    4. Optionally stems tokens (preserving negation markers).
    """
    text = fix_encoding(text)
    tokens = tokenize(text)

    # Filter stop words while strictly preserving negation terms
    filtered = [
        t for t in tokens
        if (t in NEGATION_WORDS or (t not in SAFE_STOP_WORDS and len(t) > 2))
    ]

    if preserve_stem:
        processed = [t if t in NEGATION_WORDS else stem(t) for t in filtered]
    else:
        processed = filtered

    return " ".join(processed)
