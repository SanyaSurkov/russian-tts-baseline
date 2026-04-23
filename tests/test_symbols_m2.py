# tests/test_symbols_m2.py
import pytest
from text.symbols import symbols


def test_stressed_vowels_in_inventory():
    """After M2 changes: each of 6 stressed vowel tokens must exist."""
    expected = ["@a+", "@e+", "@i+", "@o+", "@u+", "@ɨ+"]
    for tok in expected:
        assert tok in symbols, f"{tok} missing from symbols"


def test_stressed_and_unstressed_have_different_ids():
    """Each stressed vowel must have a unique id different from its unstressed version."""
    pairs = [("@a", "@a+"), ("@e", "@e+"), ("@i", "@i+"),
             ("@o", "@o+"), ("@u", "@u+"), ("@ɨ", "@ɨ+")]
    for base, stressed in pairs:
        assert base in symbols
        assert stressed in symbols
        assert symbols.index(base) != symbols.index(stressed)


def test_vocab_size_grew_by_6():
    """The new vocab should contain exactly 6 more tokens than before M2.

    Before M2: 490 tokens. After M2: 496.
    """
    assert len(symbols) == 496
