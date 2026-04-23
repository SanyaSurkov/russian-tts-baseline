# tests/test_stress_textgrid.py
import os
import tempfile
import pytest
import tgt

from russian_frontend.stress_textgrid import (
    process_textgrid,
    STRESSED_VOWEL_MAP,
)


def _make_synthetic_textgrid(tmpdir):
    """Build a minimal TextGrid with 2 words, 5 phones."""
    tg = tgt.core.TextGrid()

    # Word tier: "дом стол" with times [0.0, 0.3] and [0.3, 0.7]
    word_tier = tgt.core.IntervalTier(start_time=0.0, end_time=0.7, name="words")
    word_tier.add_interval(tgt.core.Interval(0.0, 0.3, "дом"))
    word_tier.add_interval(tgt.core.Interval(0.3, 0.7, "стол"))

    # Phone tier: д о м   с т о л
    phone_tier = tgt.core.IntervalTier(start_time=0.0, end_time=0.7, name="phones")
    phone_tier.add_interval(tgt.core.Interval(0.00, 0.10, "d"))
    phone_tier.add_interval(tgt.core.Interval(0.10, 0.20, "o"))
    phone_tier.add_interval(tgt.core.Interval(0.20, 0.30, "m"))
    phone_tier.add_interval(tgt.core.Interval(0.30, 0.40, "s"))
    phone_tier.add_interval(tgt.core.Interval(0.40, 0.50, "t"))
    phone_tier.add_interval(tgt.core.Interval(0.50, 0.60, "o"))
    phone_tier.add_interval(tgt.core.Interval(0.60, 0.70, "l"))

    tg.add_tier(word_tier)
    tg.add_tier(phone_tier)

    path = os.path.join(tmpdir, "test.TextGrid")
    tgt.io.write_to_file(tg, path, format="long")
    return path


def test_process_textgrid_renames_stressed_vowels(tmp_path):
    """Post-processing: 'до+м' and 'сто+л' → 'o' becomes 'o+' in both words."""
    tg_path = _make_synthetic_textgrid(str(tmp_path))
    out_path = str(tmp_path / "out.TextGrid")

    # Stressed lab maps word → vowel_index (0-based among word vowels)
    stress_map = {
        ("дом", 0): 0,   # first occurrence of "дом" at index 0, stress on vowel 0 (о)
        ("стол", 1): 0,  # second word at index 1, stress on vowel 0 (о)
    }

    result = process_textgrid(tg_path, stress_map, out_path)
    assert result["ok"] == 2
    assert result["skipped"] == 0
    assert result["mismatch"] == 0

    # Reload and check labels
    tg = tgt.io.read_textgrid(out_path)
    phones = [iv.text for iv in tg.get_tier_by_name("phones").intervals]
    # Original: d o m s t o l
    # Expected: d o+ m s t o+ l
    assert phones == ["d", "o+", "m", "s", "t", "o+", "l"]


def test_stressed_vowel_map_has_all_six():
    assert set(STRESSED_VOWEL_MAP.keys()) == {"a", "e", "i", "o", "u", "ɨ"}
    assert STRESSED_VOWEL_MAP["o"] == "o+"
