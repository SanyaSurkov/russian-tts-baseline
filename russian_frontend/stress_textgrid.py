# russian_frontend/stress_textgrid.py
"""Post-process M1 TextGrids: rename stressed vowel labels 'o' → 'o+' etc.

Input: existing M1 TextGrid + pre-computed stress map {(word, word_index): vowel_index}.
Output: new TextGrid with same timings but vowel at stressed position relabeled.

The stress_map is built from the stressed .lab file (output of accent.py applied
to raw .lab). See scripts/post_process_textgrids.py for the orchestration.
"""
import os
import re
from typing import Dict, Tuple

import tgt


# Base MFA vowel labels → their stressed variants.
# Allophones (ɪ, ə, ʊ, ɛ, ʉ) are unstressed by MFA's phonology and don't need + variants.
STRESSED_VOWEL_MAP = {
    "a": "a+",
    "e": "e+",
    "i": "i+",
    "o": "o+",
    "u": "u+",
    "ɨ": "ɨ+",
}


def process_textgrid(
    input_path: str,
    stress_map: Dict[Tuple[str, int], int],
    output_path: str,
) -> dict:
    """Rewrite TextGrid with stressed vowel labels.

    Args:
        input_path: M1 TextGrid path.
        stress_map: dict from (orthographic_word, word_index_in_tier) to 0-based
                    vowel index to mark as stressed.
        output_path: where to write the new TextGrid.

    Returns:
        dict with counts: {"ok": N, "skipped": N, "mismatch": N}.
        - ok: word successfully re-labeled
        - skipped: word not in stress_map (e.g. sp/sil)
        - mismatch: vowel_index out of range for this word's phones
    """
    tg = tgt.io.read_textgrid(input_path)
    word_tier = tg.get_tier_by_name("words")
    phone_tier = tg.get_tier_by_name("phones")

    ok = 0
    skipped = 0
    mismatch = 0

    for word_idx, word_interval in enumerate(word_tier.intervals):
        word_text = word_interval.text.strip()
        key = (word_text, word_idx)
        if key not in stress_map:
            skipped += 1
            continue

        target_vowel_idx = stress_map[key]

        # Find phones inside this word's time range
        word_phones = [
            ph for ph in phone_tier.intervals
            if ph.start_time >= word_interval.start_time
            and ph.end_time <= word_interval.end_time
        ]

        # Find the N-th vowel among those phones
        vowel_positions = [
            i for i, ph in enumerate(word_phones)
            if ph.text.strip() in STRESSED_VOWEL_MAP
        ]

        if target_vowel_idx >= len(vowel_positions):
            mismatch += 1
            continue

        target_phone = word_phones[vowel_positions[target_vowel_idx]]
        base_label = target_phone.text.strip()
        target_phone.text = STRESSED_VOWEL_MAP[base_label]
        ok += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tgt.io.write_to_file(tg, output_path, format="long")
    return {"ok": ok, "skipped": skipped, "mismatch": mismatch}
