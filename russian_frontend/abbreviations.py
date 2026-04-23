# russian_frontend/abbreviations.py
"""Russian abbreviation expansions for inference-time text normalization.

ABBREVIATIONS: dict from abbreviation form to spoken form.
Applied via regex with word boundaries in normalize.normalize().

Rules:
- Keys with '.' must be escaped when building regex (handled by normalize.py).
- Spoken form is lowercase (normalize.py lowercases the whole text anyway).
- Order of application: abbreviations first, then numbers, then yo-restore.
"""

ABBREVIATIONS = {
    # Университеты и учреждения
    "МГУ": "эмгэу",
    "СПбГУ": "эс-пэ-бэ-гэу",
    "МФТИ": "эм-фэ-тэ-и",
    "НИИ": "нии",
    "РАН": "ран",
    "ЦРУ": "цэ-эр-у",
    "ФСБ": "фэ-эс-бэ",
    "МВД": "эм-вэ-дэ",
    "МИД": "мид",
    "ООН": "о-о-эн",

    # Стандарты
    "ГОСТ": "гост",
    "СНиП": "эс-эн-и-пэ",

    # Компании / СМИ
    "РЖД": "эр-жэ-дэ",
    "СССР": "эс-эс-эс-эр",
    "РФ": "эр-эф",

    # Сокращения в адресах (с точкой)
    "ул.": "улица",
    "пр.": "проспект",
    "пер.": "переулок",
    "пл.": "площадь",
    "д.": "дом",
    "корп.": "корпус",
    "кв.": "квартира",
    "обл.": "область",
    "р-н": "район",

    # Сокращения в текстах
    "т.е.": "то есть",
    "т.д.": "так далее",
    "т.п.": "тому подобное",
    "т.к.": "так как",
    "и т.д.": "и так далее",
    "и т.п.": "и тому подобное",
    "др.": "другие",
    "проч.": "прочее",
    "см.": "смотри",
    "стр.": "страница",

    # Временные
    # "г." omitted: conflicts with "год"/"город" — disambiguation is future work
    "гг.": "годы",
    "в.": "век",
    "вв.": "века",

    # Единицы (опционально, может конфликтовать с контекстом)
    "кг": "килограмм",
    "км": "километр",
    "см": "сантиметр",
    "мм": "миллиметр",
    "мл": "миллилитр",
    "м.": "метр",
    "руб.": "рубль",
    "коп.": "копейка",
    "%": "процент",

    # Обращения
    "г-н": "господин",
    "г-жа": "госпожа",
}

# Lowercase all keys: normalize() does text.lower() first, so matching is case-insensitive.
ABBREVIATIONS = {k.lower(): v for k, v in ABBREVIATIONS.items()}
