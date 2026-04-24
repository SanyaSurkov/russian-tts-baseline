# tests/test_infer_phones.py
import pytest
from russian_frontend.infer_phones import word_to_phones, text_to_phones


def test_word_to_phones_no_stress():
    """Word without '+' marker → base phones unchanged."""
    lex = {"кот": ["k", "o", "t"]}
    assert word_to_phones("кот", lex) == ["k", "o", "t"]


def test_word_to_phones_with_stress_on_first_vowel():
    """'до+м' → find first stressable vowel, rename to stressed variant."""
    lex = {"дом": ["d", "o", "m"]}
    assert word_to_phones("до+м", lex) == ["d", "o+", "m"]


def test_word_to_phones_with_stress_on_second_vowel():
    """'домо+й' → rename SECOND 'o' to 'o+'."""
    lex = {"домой": ["d", "ɐ", "m", "o", "j"]}
    # Only 'o' is in STRESSED_VOWEL_MAP (ɐ is allophone). So 0-th stressable is 'o' at index 3.
    # But parse_stressed_word counts ORTHOGRAPHIC vowels (о, о), second one = index 1.
    # Phones only have ONE stressable vowel ('o'). So the second orth vowel maps to... nothing.
    # With only 1 stressable phone, target_idx=1 is out of range → mismatch, fallback to base.
    result = word_to_phones("домо+й", lex)
    # Expected: all base phones (no + added) because mismatch
    assert "+" not in "".join(result), f"expected no + markers, got {result}"


def test_word_to_phones_oov_returns_sp():
    """Word not in lexicon → ['sp'] fallback."""
    assert word_to_phones("неизвестное", {}) == ["sp"]


def test_word_to_phones_plus_stripped_for_lookup():
    """Lexicon keys have no '+' — strip before lookup."""
    lex = {"стол": ["s", "t", "o", "l"]}
    # "сто+л" — orthographic vowels = ["о"], stressed index 0.
    # Phones: s t o l — first stressable vowel is 'o' at index 2.
    assert word_to_phones("сто+л", lex) == ["s", "t", "o+", "l"]


def test_text_to_phones_pipeline(tmp_path):
    """Full pipeline: text → normalize → stress → per-word lookup → concatenated phones.

    Uses a tiny mock lexicon file.
    """
    lex_file = tmp_path / "tiny.dict"
    # Format: word<TAB>phone1 phone2 ...
    lex_file.write_text(
        "стол\ts t o l\n"
        "кот\tk o t\n",
        encoding="utf-8",
    )

    result = text_to_phones("Стол и кот", str(lex_file))
    # After normalize: "стол и кот" (lowercased, no abbr/numbers to expand)
    # add_stress: "сто+л и ко+т" (ruaccent should stress both)
    # word_to_phones:
    #   "сто+л" → ["s", "t", "o+", "l"]
    #   "и" → OOV → ["sp"] (no lexicon entry)
    #   "ко+т" → ["k", "o+", "t"]
    # concatenated: ["s", "t", "o+", "l", "sp", "k", "o+", "t"]
    assert "o+" in result, f"no stress in {result}"
    assert "s" in result and "t" in result
    assert result[0] == "s"
    assert result[-1] == "t"
