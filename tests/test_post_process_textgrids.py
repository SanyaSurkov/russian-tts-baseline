# tests/test_post_process_textgrids.py
"""Tests for build_stress_map normalisation (must match MFA word-tier)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.post_process_textgrids import build_stress_map


def test_build_stress_map_lowercase_and_yo():
    """MFA collapses ё→е and lowercases. stress_map keys must match."""
    # "Поч+ём зн+ать" → TG has ["почем", "знать"]
    m = build_stress_map("Поч+ём зн+ать")
    assert ("почем", 0) in m
    assert ("знать", 1) in m


def test_build_stress_map_hyphen_splits_compound():
    """MFA splits 'какие-нибудь' into 2 intervals. build_stress_map must match."""
    m = build_stress_map("как+ие-ниб+удь случ+ай")
    # After hyphen split: [какие(stress on и=idx1), нибудь(stress on у=idx1), случай(stress on у=idx1)]
    assert ("какие", 0) in m
    assert ("нибудь", 1) in m
    assert ("случай", 2) in m


def test_build_stress_map_strips_punctuation():
    """Commas/periods stripped; word still keyed correctly."""
    m = build_stress_map("зн+ать, м+ожет")
    assert ("знать", 0) in m
    assert ("может", 1) in m


def test_build_stress_map_unstressed_word_reserves_index():
    """Word without '+' still occupies a tg_idx slot so later words align."""
    # "а когд+а я" → TG has ["а", "когда", "я"], but only middle has stress
    m = build_stress_map("а когд+а я")
    # "когда" should be at index 1, not 0
    assert ("когда", 1) in m
    # "а" has no stress → no entry, but index 0 reserved
    assert ("а", 0) not in m
    # "я" at index 2 also no stress
    assert ("я", 2) not in m


def test_build_stress_map_hyphen_preserves_later_indices():
    """A hyphen-split in the middle must not desynchronise downstream words."""
    # 4 whitespace tokens, middle one hyphenated → 5 MFA intervals
    m = build_stress_map("пр+ивет как+ие-ниб+удь д+ела т+оже")
    assert ("привет", 0) in m
    assert ("какие", 1) in m
    assert ("нибудь", 2) in m
    assert ("дела", 3) in m
    assert ("тоже", 4) in m
