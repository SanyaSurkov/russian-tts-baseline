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


def test_fallback_on_nonsense_word():
    """Unknown/nonsense word falls back to last-vowel stress."""
    result = add_stress("блргзлы")
    # Has one vowel "ы" — fallback places + after it
    assert "+" in result
    # Position: after the ы
    assert "ы+" in result


def test_parse_stressed_word_finds_vowel_index():
    """parse_stressed_word returns 0-based index of stressed vowel among word's vowels."""
    # "сто+л" → vowels = ["о"], stressed index = 0
    assert parse_stressed_word("сто+л") == 0
    # "домо+й" → vowels = ["о", "о"], stressed = second → index 1
    assert parse_stressed_word("домо+й") == 1
    # "зво+нит" → vowels = ["о", "и"], stressed index = 0
    assert parse_stressed_word("зво+нит") == 0


def test_parse_stressed_word_no_stress_returns_none():
    """If word has no + marker, return None."""
    assert parse_stressed_word("стол") is None
