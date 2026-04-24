# scripts/validate_m2_alignment.py
"""Sanity-check M2 TextGrids after post-processing.

Samples `--sample_size` TextGrids at random and verifies:
  - at least some phones have '+' markers (stress actually inserted)
  - phone count per TextGrid matches its M1 counterpart (no drops)
  - vowel tokens are from the known inventory (a/e/i/o/u/ɨ ± '+', plus allophones)

Usage:
    python scripts/validate_m2_alignment.py \
        --m1_tg_path ./preprocessed/MULTISPK_RU/TextGrid \
        --m2_tg_path ./preprocessed/MULTISPK_RU_M2/TextGrid \
        --sample_size 100
"""
import argparse
import os
import random
from collections import Counter

import tgt


ALLOWED_VOWELS = {
    "a", "e", "i", "o", "u", "ɨ",
    "a+", "e+", "i+", "o+", "u+", "ɨ+",
    "ɐ", "ə", "ɪ", "ʊ", "ɛ", "ʉ",  # unstressed allophones (pass-through)
}


def collect_phones(tg_path):
    tg = tgt.io.read_textgrid(tg_path)
    tier = tg.get_tier_by_name("phones")
    return [iv.text.strip() for iv in tier.intervals]


def iter_textgrids(root):
    for speaker in os.listdir(root):
        spk_dir = os.path.join(root, speaker)
        if not os.path.isdir(spk_dir):
            continue
        for dirpath, _, files in os.walk(spk_dir):
            for f in files:
                if f.endswith(".TextGrid"):
                    yield os.path.join(dirpath, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_tg_path", default="./preprocessed/MULTISPK_RU/TextGrid")
    ap.add_argument("--m2_tg_path", required=True)
    ap.add_argument("--sample_size", type=int, default=100)
    args = ap.parse_args()

    m2_paths = list(iter_textgrids(args.m2_tg_path))
    if not m2_paths:
        raise SystemExit(f"no TextGrids under {args.m2_tg_path}")

    random.seed(42)
    sample = random.sample(m2_paths, k=min(args.sample_size, len(m2_paths)))
    print(f"Validating {len(sample)} / {len(m2_paths)} M2 TextGrids")

    n_with_plus = 0
    n_count_mismatch = 0
    unknown_labels = Counter()
    total_plus = 0

    for m2_path in sample:
        m2_phones = collect_phones(m2_path)
        if any("+" in p for p in m2_phones):
            n_with_plus += 1
        total_plus += sum(1 for p in m2_phones if "+" in p)
        for p in m2_phones:
            if p and p not in ALLOWED_VOWELS and not p.isalpha() and p not in {"sp", "spn", "sil", ""}:
                unknown_labels[p] += 1

        # Count-match vs M1
        rel = os.path.relpath(m2_path, args.m2_tg_path)
        m1_path = os.path.join(args.m1_tg_path, rel)
        if os.path.exists(m1_path):
            m1_phones = collect_phones(m1_path)
            if len(m1_phones) != len(m2_phones):
                n_count_mismatch += 1

    print(f"TextGrids with >=1 '+' marker: {n_with_plus}/{len(sample)}")
    print(f"Total '+' vowels in sample:   {total_plus}")
    print(f"Phone-count mismatches vs M1: {n_count_mismatch}")
    if unknown_labels:
        print(f"Unexpected labels encountered:")
        for lbl, cnt in unknown_labels.most_common(20):
            print(f"  {lbl!r}\t{cnt}")

    # Hard fail conditions
    if n_with_plus < 0.5 * len(sample):
        raise SystemExit("FAIL: <50% sampled TextGrids have any '+' marker — stress pipeline likely broken")
    if n_count_mismatch > 0:
        raise SystemExit(f"FAIL: {n_count_mismatch} TextGrids have different phone count vs M1 — post-processing altered timings")
    print("OK")


if __name__ == "__main__":
    main()
