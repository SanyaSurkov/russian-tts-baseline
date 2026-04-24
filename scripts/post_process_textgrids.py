# scripts/post_process_textgrids.py
"""Build M2 TextGrids from M1 TextGrids + stressed .lab files.

For each M1 TextGrid:
  1. Read the corresponding .lab.stressed (same basename).
  2. For each word in the .lab.stressed, compute (word, word_index) → vowel_index
     where vowel_index = position of + marker among the word's vowels.
  3. Call russian_frontend.stress_textgrid.process_textgrid() to rewrite labels.

Usage:
    python scripts/post_process_textgrids.py \
        --raw_path ./data \
        --m1_tg_path ./preprocessed/MULTISPK_RU/TextGrid \
        --m2_tg_path ./preprocessed/MULTISPK_RU_M2/TextGrid
"""
import argparse
import os
import re
from typing import Optional
from tqdm import tqdm

from russian_frontend.accent import parse_stressed_word
from russian_frontend.stress_textgrid import process_textgrid


_WORD_RE = re.compile(r"\S+")
_PUNCT_STRIP = re.compile(r"[^\w\+\-]", flags=re.UNICODE)


def build_stress_map(stressed_text: str) -> dict:
    """Parse stressed .lab content, produce {(word_no_plus, word_idx): vowel_idx}.

    MFA word-tier strips most punctuation but keeps hyphens in compound words
    like "по-русски". We mirror that: strip everything except letters/digits/
    underscore/'+'/'-'.
    """
    words = _WORD_RE.findall(stressed_text)
    stress_map = {}
    for idx, w in enumerate(words):
        vowel_idx = parse_stressed_word(w)
        if vowel_idx is None:
            continue
        clean = _PUNCT_STRIP.sub("", w).replace("+", "")
        if not clean:
            continue
        stress_map[(clean, idx)] = vowel_idx
    return stress_map


def find_stressed_lab(raw_path: str, speaker: str, basename: str, subdir: str = "") -> Optional[str]:
    """Locate the .lab.stressed file for a given TextGrid basename."""
    if subdir:
        candidate = os.path.join(raw_path, speaker, subdir, f"{basename}.lab.stressed")
    else:
        candidate = os.path.join(raw_path, speaker, f"{basename}.lab.stressed")
    return candidate if os.path.exists(candidate) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", required=True)
    ap.add_argument("--m1_tg_path", required=True)
    ap.add_argument("--m2_tg_path", required=True)
    args = ap.parse_args()

    total_tg = 0
    total_processed = 0
    total_ok = 0
    total_skipped = 0
    total_mismatch = 0
    missing_lab = 0

    speaker_dirs = [d for d in os.listdir(args.m1_tg_path)
                    if os.path.isdir(os.path.join(args.m1_tg_path, d))]
    print(f"Speakers: {speaker_dirs}")

    for speaker in speaker_dirs:
        speaker_tg_dir = os.path.join(args.m1_tg_path, speaker)
        for root, _, files in os.walk(speaker_tg_dir):
            for f in files:
                if not f.endswith(".TextGrid"):
                    continue
                total_tg += 1
                tg_path = os.path.join(root, f)
                basename = os.path.splitext(f)[0]

                rel_dir = os.path.relpath(root, speaker_tg_dir)
                subdir = "" if rel_dir == "." else rel_dir

                lab_path = find_stressed_lab(args.raw_path, speaker, basename, subdir)
                if lab_path is None:
                    missing_lab += 1
                    continue

                with open(lab_path, encoding="utf-8") as lf:
                    stressed_text = lf.readline().strip()

                stress_map = build_stress_map(stressed_text)

                if subdir:
                    out_path = os.path.join(args.m2_tg_path, speaker, subdir, f)
                else:
                    out_path = os.path.join(args.m2_tg_path, speaker, f)

                result = process_textgrid(tg_path, stress_map, out_path)
                total_processed += 1
                total_ok += result["ok"]
                total_skipped += result["skipped"]
                total_mismatch += result["mismatch"]

    print("\n=== POST-PROCESS SUMMARY ===")
    print(f"Total M1 TextGrids: {total_tg}")
    print(f"Processed: {total_processed}")
    print(f"Missing .lab.stressed: {missing_lab}")
    print(f"Word re-labels: ok={total_ok}, skipped={total_skipped}, mismatch={total_mismatch}")


if __name__ == "__main__":
    main()
