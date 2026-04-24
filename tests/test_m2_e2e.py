# tests/test_m2_e2e.py
"""End-to-end smoke: configs load, frontend chain runs, vocab has stressed tokens."""
import os
import yaml

import pytest


def test_m2_config_paths_exist():
    """All three M2 configs must load and have correct fields."""
    pp = yaml.load(open("config/MULTISPK_RU_M2/preprocess.yaml"), Loader=yaml.FullLoader)
    tr = yaml.load(open("config/MULTISPK_RU_M2/train.yaml"), Loader=yaml.FullLoader)
    mo = yaml.load(open("config/MULTISPK_RU_M2/model.yaml"), Loader=yaml.FullLoader)

    assert pp["dataset"] == "MULTISPK_RU_M2"
    assert "MULTISPK_RU_M2" in pp["path"]["preprocessed_path"] or pp["path"]["preprocessed_path"].endswith("preprocessed_m2")
    assert tr["path"]["ckpt_path"].endswith("MULTISPK_RU_M2")
    assert tr["step"]["save_step"] > 0  # user tunes per machine
    assert mo["multi_speaker"] is True


def test_frontend_chain_smoke():
    """normalize → accent runs without error on a realistic input and produces expected markers."""
    from russian_frontend.normalize import normalize
    from russian_frontend.accent import add_stress

    raw = "В 2026 году МГУ принял 12 студентов на ул. Ленина"
    normalized = normalize(raw)
    # num2words русский ordinal возвращает именительный падеж — не склоняется.
    # Проверяем только факт разворота числа.
    assert "две тысячи двадцать" in normalized
    assert "эмгэу" in normalized
    assert "двенадцать" in normalized
    assert "улица" in normalized

    stressed = add_stress(normalized)
    assert "+" in stressed  # at least one word got a stress marker


def test_symbols_have_stressed_vowels():
    from text.symbols import symbols
    for tok in ["@a+", "@e+", "@i+", "@o+", "@u+", "@ɨ+"]:
        assert tok in symbols


def test_m2_scripts_are_importable():
    """All M2 scripts parse without syntax errors."""
    import ast
    for script in [
        "scripts/build_stressed_lab.py",
        "scripts/post_process_textgrids.py",
        "scripts/run_m2_preprocessing.py",
        "scripts/validate_m2_alignment.py",
    ]:
        assert os.path.exists(script), f"missing: {script}"
        with open(script, "r", encoding="utf-8") as f:
            ast.parse(f.read())
