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


def test_exactly_six_plus_tokens_added():
    """Relative check: exactly 6 '+'-tokens in the vocab and nothing else.

    Robust to upstream cmudict size changes — we only care that our 6 new
    stressed variants are present and no extras leaked in.
    """
    plus_tokens = [s for s in symbols if "+" in s]
    assert len(plus_tokens) == 6, f"expected 6 '+'-tokens, got {plus_tokens}"
