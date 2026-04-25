# russian_frontend/infer_phones.py
"""Inference-time text → phoneme sequence, with stress markers.

Mirrors the training-time pipeline (MFA-aligned TextGrids → +-marked phones)
so synthesize.py and the omograph evaluation feed the model the same phones
that it was trained on. The MFA Russian lexicon is omograph-aware (multiple
phoneme variants per orthographic word, e.g. за́мок vs замо́к); we use ruaccent's
stress position to pick the right variant and mark its stressed vowel.
"""
import re
from typing import Dict, List, Optional

from russian_frontend.accent import add_stress, parse_stressed_word
from russian_frontend.normalize import normalize
from russian_frontend.stress_textgrid import STRESSED_VOWEL_MAP


_WORD_RE = re.compile(r"[а-яё+\-]+", re.IGNORECASE)

# Vowel-like phones in the MFA Russian inventory (full + reduced/allophones).
# Used for ORTHOGRAPHIC vowel counting — e.g. "замок" has 2 orth vowels, and
# the variant [z̪ ɐ m o k] also has 2 vowel-like phones (ɐ, o), so we can
# locate the N-th orthographic vowel in the phone sequence.
ALL_VOWELS = set(STRESSED_VOWEL_MAP.keys()) | {"ɐ", "ə", "ɪ", "ʊ", "ɵ", "ɛ", "æ", "ʉ"}


def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_lexicon(path: str) -> Dict[str, List[List[str]]]:
    """Parse MFA-style lexicon. Format per line:
        word [prob1 prob2 ...] phone1 phone2 ...

    Numeric tokens between word and phones are MFA metadata (probability /
    duration features) and MUST be filtered out — they are NOT phonemes.

    Returns a dict mapping word → list of phoneme-list variants. Multiple
    entries with the same word (e.g. omographs) are accumulated as variants.
    """
    lex: Dict[str, List[List[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.split()
            if len(toks) < 2:
                continue
            word = toks[0].lower()
            phones = [t for t in toks[1:] if not _is_float(t)]
            if not phones:
                continue
            lex.setdefault(word, []).append(phones)
    return lex


def _orth_position_of_vowel(phones: List[str], idx: int) -> int:
    """How many vowel-like phones precede phones[idx] (0-based orth position)."""
    return sum(1 for i in range(idx) if phones[i] in ALL_VOWELS)


def _try_mark_variant(variant: List[str], vowel_idx: int) -> Optional[List[str]]:
    """Locate the vowel_idx-th vowel-like phone in `variant`. If it is a
    full vowel from STRESSED_VOWEL_MAP, return a copy with that phone marked
    '+'. If it is reduced (ɐ/ə/...) — variant cannot represent the requested
    stress position, return None so the caller can try another variant.
    """
    counter = 0
    for i, p in enumerate(variant):
        if p in ALL_VOWELS:
            if counter == vowel_idx:
                if p in STRESSED_VOWEL_MAP:
                    out = list(variant)
                    out[i] = STRESSED_VOWEL_MAP[p]
                    return out
                return None
            counter += 1
    return None


def word_to_phones(stressed_word: str, lexicon: Dict[str, List[List[str]]]) -> List[str]:
    """Map one stressed orthographic token to a phoneme list with stress marker.

    Algorithm:
      1. Strip '+' to get raw word; OOV → ['sp'].
      2. If no '+' marker → return first lexicon variant unchanged.
      3. Among lexicon variants, pick the one whose N-th orthographic vowel
         is a FULL vowel (a/e/i/o/u/ɨ); mark it with '+'.
      4. If no variant has a full vowel at that position (e.g. a single-variant
         entry where MFA stored a reduced phone), return the first variant
         unchanged — matches training behaviour, where post_process_textgrids
         also could not mark such phones.
    """
    clean = stressed_word.replace("+", "").lower()
    if clean not in lexicon:
        return ["sp"]

    variants = lexicon[clean]
    vowel_idx = parse_stressed_word(stressed_word)

    if vowel_idx is None:
        return list(variants[0])

    for variant in variants:
        marked = _try_mark_variant(variant, vowel_idx)
        if marked is not None:
            return marked

    return list(variants[0])


def text_to_phones(text: str, lexicon_path: str) -> List[str]:
    """Full pipeline: raw text → normalized → stressed → phoneme list.

    Hyphenated compounds (e.g. 'какие-нибудь') are split because MFA splits
    them into separate intervals during alignment, and training phones reflect
    that split.
    """
    lexicon = read_lexicon(lexicon_path)
    normalized = normalize(text)
    stressed = add_stress(normalized)

    phones: List[str] = []
    for word in _WORD_RE.findall(stressed):
        for sw in word.split("-"):
            if sw:
                phones.extend(word_to_phones(sw, lexicon))
    return phones
