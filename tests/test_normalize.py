# tests/test_normalize.py
import pytest
from russian_frontend.normalize import normalize


@pytest.mark.parametrize("raw,expected", [
    # Числа (кардинальные — num2words стабилен)
    ("12 домов", "двенадцать домов"),
    ("в 5 часов", "в пять часов"),
    ("было 100 человек", "было сто человек"),

    # Аббревиатуры
    ("МГУ", "эмгэу"),
    ("ГОСТ", "гост"),
    ("ул. Ленина", "улица ленина"),
    ("д. 5", "дом пять"),
    ("т.е. так", "то есть так"),

    # Ё-restore
    ("все знают", "всё знают"),
    ("нашел клад", "нашёл клад"),

    # Комбинации кардинальных
    ("ул. Пушкина д. 10", "улица пушкина дом десять"),

    # Lowercase
    ("Привет Мир", "привет мир"),

    # Нет замен — идентично
    ("обычное предложение", "обычное предложение"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw,fragments", [
    # Порядковые: num2words возвращает именительный падеж, проверяем substring
    ("в 2026 году", ["тысячи", "двадцать", "шест", "году"]),
    ("20 век", ["двадцат", "век"]),
    ("в 1941 году началась", ["тысяча", "девятьсот", "сорок", "перв", "году", "началась"]),
])
def test_normalize_ordinal_fragments(raw, fragments):
    """Ordinals have grammatical variants (шестой/шестом/шестым); check key fragments."""
    result = normalize(raw)
    for frag in fragments:
        assert frag in result, f"expected fragment '{frag}' in {result!r}"
