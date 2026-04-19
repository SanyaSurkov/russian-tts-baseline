"""Build held-out test set from preprocessed val.txt.

Samples ~target/n_speakers utterances per speaker, writes pipe-delimited lines.
"""
import argparse
import os
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", required=True, help="path to preprocessed val.txt")
    parser.add_argument(
        "--out",
        default="scripts/eval_testsets/general_heldout.txt",
        help="output path",
    )
    parser.add_argument("--target", type=int, default=200, help="target total utterances")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.val, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    groups = defaultdict(list)
    for ln in lines:
        parts = ln.split("|")
        if len(parts) < 2:
            continue
        spk = parts[1]
        groups[spk].append(ln)

    if not groups:
        raise SystemExit("no speakers found in val file")

    per_spk = args.target // len(groups)
    if per_spk < 1:
        per_spk = 1

    picked = []
    for spk, group in groups.items():
        k = min(per_spk, len(group))
        picked.extend(random.sample(group, k))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for ln in picked:
            f.write(ln + "\n")

    print(f"wrote {len(picked)} utterances")


if __name__ == "__main__":
    main()
