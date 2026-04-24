# russian_frontend/infer_phones.py
"""Inference-time text → phoneme sequence, with stress markers.

Fixes train/inference mismatch: training TextGrids have stressed vowels
labelled 'a+/e+/.../ɨ+' via post_process_textgrids, but the base
russian_mfa.dict only knows unmarked phonemes. This module bridges the gap:

  text → normalize → add_stress → per-word lookup with vowel-index-aware
  phoneme relabelling → flat phoneme list

Call from synthesize.preprocess_russian and any evaluation script that
generates model input from raw text.

Note on ruaccent format: ruaccent places '+' BEFORE the stressed vowel
(e.g. 'ст+ол', 'к+от'), whereas accent.parse_stressed_word expects '+'
AFTER the vowel ('сто+л'). This module uses _parse_vowel_idx() which
handles ruaccent's actual output format.
"""
import re
from typing import List, Dict, Optional

from russian_frontend.accent import add_stress
from russian_frontend.normalize import normalize
from russian_frontend.stress_textgrid import STRESSED_VOWEL_MAP


# Permit '+' and hyphen in word tokens (stressed/compound words).
_WORD_RE = re.compile(r"[а-яё+\-]+", re.IGNORECASE)

_RU_VOWELS = "аеёиоуыэюяaeiou"
_VOWEL_RE = re.compile(f"[{_RU_VOWELS}]", re.IGNORECASE)


def _parse_vowel_idx(word: str) -> Optional[int]:
    """Return 0-based index of the stressed vowel among word's vowels.

    Handles two '+' placement conventions:
      - ruaccent format: '+' BEFORE stressed vowel  ('ст+ол', 'к+от')
      - post-vowel format: '+' AFTER stressed vowel  ('сто+л', 'ко+т')

    Detects which convention by checking the character adjacent to '+'.
    Returns None if word has no '+' or the marker is not adjacent to a vowel.
    """
    if "+" not in word:
        return None
    plus_pos = word.index("+")

    # Determine stressed vowel position in the original string
    if plus_pos + 1 < len(word) and _VOWEL_RE.match(word[plus_pos + 1]):
        # ruaccent format: '+' before vowel → stressed char is at plus_pos+1
        # After removing '+', that char shifts to plus_pos
        stressed_char_idx_in_clean = plus_pos
    elif plus_pos > 0 and _VOWEL_RE.match(word[plus_pos - 1]):
        # post-vowel format: '+' after vowel → stressed char is at plus_pos-1
        # After removing '+', that char stays at plus_pos-1
        stressed_char_idx_in_clean = plus_pos - 1
    else:
        return None

    clean = word.replace("+", "")
    vowel_count = 0
    for i, ch in enumerate(clean):
        if _VOWEL_RE.match(ch):
            if i == stressed_char_idx_in_clean:
                return vowel_count
            vowel_count += 1
    return None


def read_lexicon(path: str) -> Dict[str, List[str]]:
    """Parse TSV-like MFA lexicon: 'word<TAB>p1 p2 ...'."""
    lex: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            word, phones_str = parts
            lex[word.lower()] = phones_str.split()
    return lex


def word_to_phones(stressed_word: str, lexicon: Dict[str, List[str]]) -> List[str]:
    """Single-word: map stressed orthographic form to stressed phoneme list.

    Steps:
      1. Strip '+' marker to get raw word → lookup in lexicon.
      2. If OOV → return ['sp'] (silence fallback).
      3. Use parse_stressed_word to find the N-th vowel that's stressed.
      4. Locate that N-th *stressable* phone in the phoneme sequence (i.e. phone
         whose label is a key of STRESSED_VOWEL_MAP). Rename to its '+' variant.
      5. If no '+' in stressed_word or N is out of range → return base phones.
    """
    clean = stressed_word.replace("+", "").lower()
    if clean not in lexicon:
        return ["sp"]

    phones = list(lexicon[clean])

    vowel_idx = _parse_vowel_idx(stressed_word)
    if vowel_idx is None:
        return phones

    stressable_positions = [i for i, p in enumerate(phones) if p in STRESSED_VOWEL_MAP]
    if vowel_idx >= len(stressable_positions):
        # Mismatch (orthographic vowel count > phoneme stressable count,
        # usually because MFA mapped the vowel to an allophone like ɪ/ə/ʊ).
        # Safer to return base phones than guess wrong.
        return phones

    target = stressable_positions[vowel_idx]
    phones[target] = STRESSED_VOWEL_MAP[phones[target]]
    return phones


def text_to_phones(text: str, lexicon_path: str) -> List[str]:
    """Full pipeline: raw text → normalized → stressed → phoneme list."""
    lexicon = read_lexicon(lexicon_path)
    normalized = normalize(text)
    stressed = add_stress(normalized)

    phones: List[str] = []
    for word in _WORD_RE.findall(stressed):
        phones.extend(word_to_phones(word, lexicon))
    return phones
