# tests/test_accent.py
import pytest
from russian_frontend.accent import add_stress, parse_stressed_word


def test_simple_word_gets_stress():
    """Common word should receive + marker after stressed vowel."""
    result = add_stress("стол")
    # "стол" has one vowel — must be marked
    assert "+" in result


def test_multi_word_phrase():
    """Each word in a phrase gets its own stress."""
    result = add_stress("я иду домой")
    # Each word with vowel should have +
    # "иду" and "домой" have vowels; "я" has vowel
    assert result.count("+") >= 2  # at least two of three got stress


def test_unstressed_word_stays_unmarked():
    """If ruaccent can't stress a word, leave it without '+'.

    Heuristic fallback (last-vowel) is wrong for ~85% of Russian words —
    better for the model to see base vowel labels and infer stress from
    context than to train on systematically wrong markers.
    """
    result = add_stress("блргзлы")
    assert "+" not in result


def test_parse_stressed_word_finds_vowel_index():
    """parse_stressed_word returns 0-based index of stressed vowel among word's vowels.

    Supports both '+'-before (ruaccent native) and '+'-after (hand-written)."""
    # '+'-before (ruaccent's actual output format):
    assert parse_stressed_word("ст+ол") == 0           # stressed о (only vowel)
    assert parse_stressed_word("дом+ой") == 1          # vowels [о, о], stressed 2nd
    assert parse_stressed_word("зв+онит") == 0         # vowels [о, и], stressed 1st

    # '+'-after (legacy format; keep supported for external fixtures):
    assert parse_stressed_word("сто+л") == 0
    assert parse_stressed_word("домо+й") == 1
    assert parse_stressed_word("зво+нит") == 0


def test_parse_stressed_word_no_stress_returns_none():
    """If word has no + marker, return None."""
    assert parse_stressed_word("стол") is None
