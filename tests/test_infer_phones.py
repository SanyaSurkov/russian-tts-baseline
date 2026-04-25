# tests/test_infer_phones.py
import pytest
from russian_frontend.infer_phones import (
    read_lexicon,
    word_to_phones,
    text_to_phones,
)


def test_word_to_phones_no_stress():
    """Word without '+' marker → first variant unchanged."""
    lex = {"кот": [["k", "o", "t"]]}
    assert word_to_phones("кот", lex) == ["k", "o", "t"]


def test_word_to_phones_with_stress_on_first_vowel():
    """'до+м' → mark first vowel, which is full → 'o+'."""
    lex = {"дом": [["d", "o", "m"]]}
    assert word_to_phones("до+м", lex) == ["d", "o+", "m"]


def test_word_to_phones_with_stress_on_second_vowel():
    """'домо+й' → 2nd ORTH vowel (the о). In phones [d ɐ m o j], the 2nd
    vowel-like phone is 'o' which IS stressable → mark 'o+'."""
    lex = {"домой": [["d", "ɐ", "m", "o", "j"]]}
    assert word_to_phones("домо+й", lex) == ["d", "ɐ", "m", "o+", "j"]


def test_word_to_phones_oov_returns_sp():
    assert word_to_phones("неизвестное", {}) == ["sp"]


def test_word_to_phones_plus_stripped_for_lookup():
    lex = {"стол": [["s̪", "t̪", "o", "ɫ"]]}
    assert word_to_phones("сто+л", lex) == ["s̪", "t̪", "o+", "ɫ"]


def test_word_to_phones_picks_correct_omograph_variant_castle():
    """за́мок (з+амок): 1st ORTH vowel stressed. Variant [z̪ a m ə k] has
    full 'a' at position 0 → choose this variant, mark a+."""
    lex = {"замок": [
        ["z̪", "a", "m", "ə", "k"],   # за́мок
        ["z̪", "ɐ", "m", "o", "k"],   # замо́к
    ]}
    assert word_to_phones("з+амок", lex) == ["z̪", "a+", "m", "ə", "k"]


def test_word_to_phones_picks_correct_omograph_variant_lock():
    """замо́к (зам+ок): 2nd ORTH vowel stressed. Variant 1 has reduced 'ə'
    at position 1 → reject. Variant 2 has full 'o' at position 1 → mark o+."""
    lex = {"замок": [
        ["z̪", "a", "m", "ə", "k"],
        ["z̪", "ɐ", "m", "o", "k"],
    ]}
    assert word_to_phones("зам+ок", lex) == ["z̪", "ɐ", "m", "o+", "k"]


def test_word_to_phones_skips_post_tonic_vowel():
    """In 'старинный' [s̪ t̪ ɐ rʲ i n̪ː ɨ j], two stressable phones (i, ɨ).
    ruaccent says stress on 'и' (2nd orth vowel, idx=1). Algorithm must mark
    the 'i' (2nd vowel-like phone), NOT the trailing 'ɨ'."""
    lex = {"старинный": [["s̪", "t̪", "ɐ", "rʲ", "i", "n̪ː", "ɨ", "j"]]}
    out = word_to_phones("стар+инный", lex)
    assert out == ["s̪", "t̪", "ɐ", "rʲ", "i+", "n̪ː", "ɨ", "j"]


def test_word_to_phones_reduced_only_variant_falls_back():
    """If lexicon has only one variant and it is fully reduced at the stressed
    position (no full vowel available), return base phones — same as training,
    where post_process_textgrids also could not mark such phones."""
    lex = {"сломан": [["s", "ɫ", "ɐ", "m", "ə", "n̪"]]}
    out = word_to_phones("сл+оман", lex)
    assert "+" not in "".join(out)
    assert out == ["s", "ɫ", "ɐ", "m", "ə", "n̪"]


def test_read_lexicon_filters_numeric_metadata(tmp_path):
    """MFA russian_mfa.dict has 4 floats per line between word and phones.
    They must be stripped — they are not phonemes."""
    p = tmp_path / "lex.dict"
    p.write_text(
        "замок\t0.42\t0.53\t1.98\t1.19\tz̪ a m ə k\n"
        "замок\t0.99\t0.69\t1.58\t1.1\tz̪ ɐ m o k\n",
        encoding="utf-8",
    )
    lex = read_lexicon(str(p))
    assert "замок" in lex
    assert len(lex["замок"]) == 2
    for variant in lex["замок"]:
        for tok in variant:
            assert not tok.replace(".", "").isdigit(), f"numeric leaked: {tok}"
    assert lex["замок"][0] == ["z̪", "a", "m", "ə", "k"]
    assert lex["замок"][1] == ["z̪", "ɐ", "m", "o", "k"]


def test_read_lexicon_accumulates_variants(tmp_path):
    p = tmp_path / "lex.dict"
    p.write_text(
        "стол\ts̪ t̪ o ɫ\n"
        "стол\ts̪ t̪ o ɫʲ\n",
        encoding="utf-8",
    )
    lex = read_lexicon(str(p))
    assert len(lex["стол"]) == 2


def test_text_to_phones_picks_omograph_by_context(tmp_path):
    """Full pipeline: ruaccent disambiguates omograph by context, infer_phones
    selects the matching lexicon variant. This is the critical end-to-end
    guarantee for the M2 omograph experiment."""
    p = tmp_path / "lex.dict"
    p.write_text(
        "замок\tz̪ a m ə k\n"
        "замок\tz̪ ɐ m o k\n"
        "старинный\ts̪ t̪ ɐ rʲ i n̪ː ɨ j\n"
        "был\tb ɨ ɫ\n"
        "сломан\ts ɫ ɐ m ə n̪\n",
        encoding="utf-8",
    )
    castle_phones = text_to_phones("Старинный замок.", str(p))
    lock_phones = text_to_phones("Замок был сломан.", str(p))
    assert "a+" in castle_phones, f"expected a+ in castle phones, got {castle_phones}"
    assert "o+" in lock_phones, f"expected o+ in lock phones, got {lock_phones}"
    assert castle_phones != lock_phones


def test_text_to_phones_pipeline(tmp_path):
    lex_file = tmp_path / "tiny.dict"
    lex_file.write_text(
        "стол\ts t o l\n"
        "кот\tk o t\n",
        encoding="utf-8",
    )
    result = text_to_phones("Стол и кот", str(lex_file))
    assert "o+" in result, f"no stress in {result}"
    assert result[0] == "s"
    assert result[-1] == "t"
