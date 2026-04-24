# russian_frontend/accent.py
"""ruaccent-based Russian stress marker.

add_stress(text) returns text with '+' inserted after each stressed vowel
(ruaccent's native format). Words ruaccent fails to stress are left unchanged
and logged to logs/unstressed_words.txt — no heuristic fallback (last-vowel
stress is wrong for ~85% of Russian words).

Downstream: post_process_textgrids.py skips any word without '+' in the
stressed .lab → phoneme labels for that word stay as base MFA labels.

Expose parse_stressed_word() to find the 0-based index of the stressed vowel
among the word's vowels — used by stress_textgrid.py.
"""
import os
import re
import threading
from typing import Optional

_RU_VOWELS = "аеёиоуыэюяaeiou"
_VOWEL_RE = re.compile(f"[{_RU_VOWELS}]", re.IGNORECASE)

_accentizer = None
_accentizer_lock = threading.Lock()

_UNSTRESSED_LOG_PATH = "logs/unstressed_words.txt"


def _get_accentizer():
    """Lazy-load ruaccent once per process (thread-safe)."""
    global _accentizer
    if _accentizer is None:
        with _accentizer_lock:
            if _accentizer is None:
                from ruaccent import RUAccent
                a = RUAccent()
                a.load(omograph_model_size="turbo", use_dictionary=True)
                _accentizer = a
    return _accentizer


def _log_unstressed(word: str) -> None:
    os.makedirs(os.path.dirname(_UNSTRESSED_LOG_PATH), exist_ok=True)
    with open(_UNSTRESSED_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(word + "\n")


def add_stress(text: str) -> str:
    """Add stress markers to Russian text using ruaccent.

    Words ruaccent can't stress are left unchanged (no heuristic fallback) and
    logged to logs/unstressed_words.txt for review.
    """
    a = _get_accentizer()
    stressed = a.process_all(text)

    tokens = re.split(r"(\W+)", stressed)
    out = []
    for tok in tokens:
        if not tok:
            out.append(tok)
            continue
        if not _VOWEL_RE.search(tok):
            out.append(tok)
            continue
        if "+" in tok:
            out.append(tok)
        else:
            _log_unstressed(tok)
            out.append(tok)
    return "".join(out)


def parse_stressed_word(word: str) -> Optional[int]:
    """Return 0-based index of the stressed vowel among word's vowels.

    Args:
        word: a single word that may contain '+' after a vowel.

    Returns:
        Index into the sequence of vowels in `word`, or None if no '+' marker.
    """
    if "+" not in word:
        return None

    plus_pos = word.index("+")
    if plus_pos == 0:
        return None
    if not _VOWEL_RE.match(word[plus_pos - 1]):
        return None

    clean = word.replace("+", "")
    stressed_char_idx = plus_pos - 1
    vowel_count = 0
    for i, ch in enumerate(clean):
        if _VOWEL_RE.match(ch):
            if i == stressed_char_idx:
                return vowel_count
            vowel_count += 1
    return None
