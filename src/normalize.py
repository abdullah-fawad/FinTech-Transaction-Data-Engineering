from __future__ import annotations
import re
import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_NUMBER_RE = re.compile(r"\b\d{2,}\b")  # standalone numeric tokens (amounts, dates, refs)
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.strip().lower()
    t = _PUNCT_RE.sub(" ", t)               # strip punctuation
    t = _TRAILING_NUMBER_RE.sub(" ", t)      # drop standalone multi-digit numbers (amounts/refs)
    t = _WHITESPACE_RE.sub(" ", t).strip()   # collapse whitespace
    return t


def normalize_series(s: pd.Series) -> pd.Series:
    return s.apply(normalize)


if __name__ == "__main__":
    samples = ["Petrol", "PETROL", "petrol 500", "Bolan petrol", "  Netflix   May ", "raj grocery", "cash from Haris 2000"]
    for s in samples:
        print(f"{s!r:30s} -> {normalize(s)!r}")
