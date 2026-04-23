# russian_frontend/normalize.py
"""Inference-time text normalization for Russian TTS.

Pipeline:
  1. Lowercase
  2. Abbreviations (longest keys first)
  3. Numbers and dates (num2words)
  4. Yo-restore (dictionary-based)

Use normalize(text) from eval scripts and synthesis entry points.
Training data is NOT normalized — M1 TextGrids align on raw text.
"""
import re
from num2words import num2words

from russian_frontend.abbreviations import ABBREVIATIONS
from russian_frontend.yo_words import YO_REPLACEMENTS


# Compile regex patterns once
_YEAR_CONTEXT = re.compile(r"(\d+)\s+(год[уае]?|века?|веко[въ])")
_CENTURY_CONTEXT = re.compile(r"(\d+)\s+век")
_STANDALONE_NUMBER = re.compile(r"\b(\d+)\b")

# Abbreviations: longest keys first to match "и т.д." before "т.д."
_ABBR_PATTERNS = sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0]))


def _expand_abbreviations(text: str) -> str:
    """Apply ABBREVIATIONS dict to text with word-boundary-aware matching.

    Keys with '.' are matched literally (no word boundary after).
    Ordering: longest first.
    """
    for abbr, expansion in _ABBR_PATTERNS:
        escaped = re.escape(abbr)
        if abbr.endswith(".") or abbr.endswith("-"):
            pattern = r"(?<!\w)" + escaped
        else:
            pattern = r"\b" + escaped + r"\b"
        text = re.sub(pattern, expansion, text)
    return text


def _expand_numbers(text: str) -> str:
    """Expand numeric tokens. Ordinal if followed by год/век, else cardinal."""
    def _ordinal_year(m):
        num = int(m.group(1))
        ordinal = num2words(num, lang="ru", to="ordinal")
        return f"{ordinal} {m.group(2)}"

    text = _YEAR_CONTEXT.sub(_ordinal_year, text)

    def _cardinal(m):
        num = int(m.group(1))
        return num2words(num, lang="ru")

    text = _STANDALONE_NUMBER.sub(_cardinal, text)
    return text


def _restore_yo(text: str) -> str:
    """Replace е → ё only for whitelisted words (YO_REPLACEMENTS dict)."""
    tokens = re.split(r"(\W+)", text)
    out = []
    for tok in tokens:
        lowered = tok.lower()
        if lowered in YO_REPLACEMENTS:
            out.append(YO_REPLACEMENTS[lowered])
        else:
            out.append(tok)
    return "".join(out)


def normalize(text: str) -> str:
    """Normalize Russian text for TTS inference."""
    text = text.lower()
    text = _expand_abbreviations(text)
    text = _expand_numbers(text)
    text = _restore_yo(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
