"""One-off: probe WHY Stage 2 skipped/mismatched 80% of words.

Compares .lab.stressed whitespace-tokens against M1 TextGrid word-tier intervals
to pinpoint index drift (sp/sil? empty intervals? punctuation? hyphen split?).
"""
import os
import re
import tgt
from collections import Counter

M1_TG = "C:/Users/Admin/thesis/data/preprocessed/TextGrid"
RAW = "C:/Users/Admin/thesis/data/unified"

samples = []
for root, _, files in os.walk(M1_TG):
    for f in files:
        if f.endswith(".TextGrid"):
            samples.append(os.path.join(root, f))
        if len(samples) >= 5:
            break
    if len(samples) >= 5:
        break

for tg_path in samples:
    rel = os.path.relpath(tg_path, M1_TG).replace(os.sep, "/")
    parts = rel.split("/")
    speaker = parts[0]
    basename = os.path.splitext(parts[-1])[0]
    subdir = "/".join(parts[1:-1])

    if subdir:
        lab_s = os.path.join(RAW, speaker, subdir, basename + ".lab.stressed")
    else:
        lab_s = os.path.join(RAW, speaker, basename + ".lab.stressed")

    if not os.path.exists(lab_s):
        print(f"MISSING .lab.stressed for {rel}")
        continue

    with open(lab_s, encoding="utf-8") as f:
        stressed = f.readline().strip()

    tg = tgt.io.read_textgrid(tg_path)
    wt = tg.get_tier_by_name("words")

    stressed_tokens = re.findall(r"\S+", stressed)
    word_intervals = wt.intervals
    non_empty = [iv for iv in word_intervals if iv.text.strip()]
    empty_count = sum(1 for iv in word_intervals if iv.text.strip() == "")
    sp_like = sum(1 for iv in word_intervals if iv.text.strip() in ("sp", "sil", "spn"))

    print(f"--- {rel} ---")
    print(f"  .lab.stressed: {stressed[:120]}")
    print(f"  stressed tokens: {len(stressed_tokens)}")
    print(f"  TG word intervals (total): {len(word_intervals)}")
    print(f"  TG word intervals (non-empty): {len(non_empty)}")
    print(f"  TG empty-text word intervals: {empty_count}")
    print(f"  TG sp/sil/spn word intervals: {sp_like}")
    print(f"  First 8 TG word labels: {[iv.text for iv in word_intervals[:8]]}")
    print(f"  First 8 stressed tokens: {stressed_tokens[:8]}")
    print()
