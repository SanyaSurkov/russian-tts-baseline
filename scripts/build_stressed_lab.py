# scripts/build_stressed_lab.py
"""Apply ruaccent to every .lab in data/, save as .lab.stressed sibling.

Usage:
    python scripts/build_stressed_lab.py \
        --raw_path ./data \
        --speaker_dirs ruslan mailabs_m01 mailabs_m02 mailabs_m03

Output: for every <path>/<basename>.lab creates <path>/<basename>.lab.stressed
with ruaccent-marked text on a single line.
"""
import argparse
import os
from tqdm import tqdm

from russian_frontend.accent import add_stress


def iter_lab_files(raw_path: str, speakers):
    for speaker in speakers:
        speaker_path = os.path.join(raw_path, speaker)
        if not os.path.isdir(speaker_path):
            print(f"skip missing: {speaker_path}")
            continue
        for root, _, files in os.walk(speaker_path):
            for f in files:
                if f.endswith(".lab"):
                    yield os.path.join(root, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", required=True)
    ap.add_argument("--speaker_dirs", nargs="+", required=True)
    args = ap.parse_args()

    lab_paths = list(iter_lab_files(args.raw_path, args.speaker_dirs))
    print(f"Found {len(lab_paths)} .lab files")

    n_ok = 0
    for path in tqdm(lab_paths):
        with open(path, "r", encoding="utf-8") as f:
            text = f.readline().strip()
        stressed = add_stress(text)
        out_path = path + ".stressed"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(stressed + "\n")
        n_ok += 1

    print(f"Wrote {n_ok} stressed .lab files")
    if os.path.exists("logs/unstressed_words.txt"):
        with open("logs/unstressed_words.txt") as f:
            n_unstressed = sum(1 for _ in f)
        print(f"Unstressed fallback applied to {n_unstressed} words (see logs/unstressed_words.txt)")


if __name__ == "__main__":
    main()
