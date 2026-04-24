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


def test_process_textgrid_handles_yo_and_case(tmp_path):
    """MFA word-tier is lowercase + ё→е; stress_map keys must match after
    defensive normalisation in process_textgrid.
    """
    tg = tgt.core.TextGrid()
    wt = tgt.core.IntervalTier(start_time=0.0, end_time=0.4, name="words")
    # Simulate MFA output: uppercase + ё present (shouldn't happen, but defensive)
    wt.add_interval(tgt.core.Interval(0.0, 0.4, "Ёлка"))
    pt = tgt.core.IntervalTier(start_time=0.0, end_time=0.4, name="phones")
    pt.add_interval(tgt.core.Interval(0.00, 0.10, "j"))
    pt.add_interval(tgt.core.Interval(0.10, 0.20, "o"))
    pt.add_interval(tgt.core.Interval(0.20, 0.30, "l"))
    pt.add_interval(tgt.core.Interval(0.30, 0.40, "k"))
    tg.add_tier(wt)
    tg.add_tier(pt)

    in_path = str(tmp_path / "yo.TextGrid")
    out_path = str(tmp_path / "yo_out.TextGrid")
    tgt.io.write_to_file(tg, in_path, format="long")

    # stress_map key uses build_stress_map convention: lowercase + ё→е
    stress_map = {("елка", 0): 0}
    result = process_textgrid(in_path, stress_map, out_path)
    assert result["ok"] == 1, result
