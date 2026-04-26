#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch synthesis of 29 phrases using synthesize.py"""

import subprocess
import sys

PHRASES = [
    "Утром солнце светило в окно.",
    "Кот спал на мягком диване.",
    "Мама варила суп на кухне.",
    "Дети играли во дворе до вечера.",
    "Автобус пришел на остановку вовремя.",
    "В магазине купили хлеб и молоко.",
    "Дождь лил всю ночь без перерыва.",
    "Соседка выгуливала собаку в парке.",
    "Врач принял пациента без очереди.",
    "Студент сдал экзамен на отлично.",
    "Бабушка пекла пироги с яблоками.",
    "Машина стояла у подъезда три дня.",
    "Почтальон принес письмо и журнал.",
    "На улице было тепло и светло.",
    "Официант принес счет и чаевые.",
    "Девочка рисовала дом и дерево.",
    "Мужчина читал газету на скамейке.",
    "В саду цвели розы и ромашки.",
    "Ребенок потерял игрушку в песке.",
    "Учитель написал задачу на доске.",
    "Вечером вся семья смотрела фильм.",
    "Повар посолил суп и попробовал.",
    "Замок дверной скрипел от ветра.",
    "Мука лежала в миске на столе.",
    "Коса блестела на солнце ржаная.",
    "Все ели суп из свежей капусты.",
    "Снег лежал на крыше толстым слоем.",
    "Замок стоял на холме сто лет.",
    "Медведь спал в берлоге всю зиму.",
]

BASE_CMD = [
    "python", "synthesize.py",
    "--restore_step", "73752",
    "--mode", "single",
    "--speaker_id", "3",
    "-p", "config/MULTISPK_RU_GAVNO/preprocess.yaml",
    "-m", "config/MULTISPK_RU_GAVNO/model.yaml",
    "-t", "config/MULTISPK_RU_GAVNO/train.yaml",
]


def synthesize_phrases():
    total = len(PHRASES)
    for i, phrase in enumerate(PHRASES, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] Synthesizing: {phrase}")
        print(f"{'='*60}")

        cmd = BASE_CMD + ["--text", phrase]

        try:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                check=True
            )
            print(f"✅ Done: phrase {i}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error on phrase {i}: {e}")
            # Continue with next phrase instead of stopping
            continue

    print(f"\n{'='*60}")
    print(f"🎉 Batch synthesis complete! {total} phrases processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    synthesize_phrases()