"""Re-split M2 train.txt/val.txt so utterances match M1's split by basename.

After preprocessor.py (Stage 3) ran for M2 with its own random shuffle, this
script shuffles entries between M2 train.txt and M2 val.txt so each utterance
goes into the same bucket it did in M1. Phone sequences remain M2's ('+'-aware).

Result: both experiments evaluate on identical utterance sets → val-loss /
MCD / WER comparisons are apples-to-apples.

Usage:
    python scripts/align_m2_split_to_m1.py \
        --m1_preprocessed C:/Users/Admin/thesis/data/preprocessed \
        --m2_preprocessed C:/Users/Admin/thesis/data/preprocessed_m2
"""
import argparse
import os
import shutil


def read_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def basename_of(entry):
    return entry.split("|", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_preprocessed", required=True)
    ap.add_argument("--m2_preprocessed", required=True)
    ap.add_argument("--verbose", action="store_true", help="print missing basenames")
    args = ap.parse_args()

    m1_train = os.path.join(args.m1_preprocessed, "train.txt")
    m1_val = os.path.join(args.m1_preprocessed, "val.txt")
    m2_train = os.path.join(args.m2_preprocessed, "train.txt")
    m2_val = os.path.join(args.m2_preprocessed, "val.txt")

    for p in (m1_train, m1_val, m2_train, m2_val):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    # Basenames from M1
    m1_train_names = {basename_of(e) for e in read_entries(m1_train)}
    m1_val_names = {basename_of(e) for e in read_entries(m1_val)}
    print(f"M1: train={len(m1_train_names)}, val={len(m1_val_names)}, total={len(m1_train_names | m1_val_names)}")

    # Basenames from M2 — build sets ONCE
    m2_train_entries = read_entries(m2_train)
    m2_val_entries = read_entries(m2_val)
    m2_train_names = {basename_of(e) for e in m2_train_entries}
    m2_val_names = {basename_of(e) for e in m2_val_entries}
    m2_all_entries = m2_train_entries + m2_val_entries
    m2_all_names = m2_train_names | m2_val_names
    print(f"M2: train={len(m2_train_entries)}, val={len(m2_val_entries)}, total={len(m2_all_entries)}")

    # --- DIAGNOSTIC: check for missing/extra entries ---
    missing_in_m2 = (m1_train_names | m1_val_names) - m2_all_names
    extra_in_m2 = m2_all_names - (m1_train_names | m1_val_names)

    if missing_in_m2:
        print(f"\n⚠️  WARN: {len(missing_in_m2)} M1 basenames MISSING from M2 preprocessed")
        if args.verbose:
            for name in sorted(missing_in_m2)[:20]:
                print(f"    - {name}")
            if len(missing_in_m2) > 20:
                print(f"    ... and {len(missing_in_m2) - 20} more")

    if extra_in_m2:
        print(f"\n⚠️  WARN: {len(extra_in_m2)} basenames in M2 but NOT in M1")
        if args.verbose:
            for name in sorted(extra_in_m2)[:20]:
                print(f"    - {name}")

    # --- RE-SPLIT: O(1) lookup per entry ---
    new_train, new_val = [], []
    train_from_old_train, train_from_old_val = 0, 0
    val_from_old_train, val_from_old_val = 0, 0

    for e in m2_all_entries:
        b = basename_of(e)
        if b in m1_val_names:
            new_val.append(e)
            if b in m2_train_names:
                val_from_old_train += 1
            else:
                val_from_old_val += 1
        else:
            new_train.append(e)
            if b in m2_train_names:
                train_from_old_train += 1
            else:
                train_from_old_val += 1

    print(f"\nRe-split result:")
    print(f"  M2 train = {len(new_train)} (from old train: {train_from_old_train}, from old val: {train_from_old_val})")
    print(f"  M2 val = {len(new_val)} (from old train: {val_from_old_train}, from old val: {val_from_old_val})")

    # Validate: no overlaps
    new_train_names = {basename_of(e) for e in new_train}
    new_val_names = {basename_of(e) for e in new_val}
    overlap = new_train_names & new_val_names
    if overlap:
        raise SystemExit(f"FATAL: {len(overlap)} basenames in BOTH train and val")

    # Coverage check
    covered = len(m1_val_names & new_val_names)
    pct = 100 * covered / len(m1_val_names) if m1_val_names else 0
    print(f"\nM1 val coverage: {covered}/{len(m1_val_names)} ({pct:.1f}%)")
    if pct < 95:
        print(f"⚠️  WARNING: Low coverage. Investigate why M2 preprocessor dropped utterances.")
    else:
        print(f"✅ OK")

    # Backup & overwrite
    shutil.copy(m2_train, m2_train + ".bak")
    shutil.copy(m2_val, m2_val + ".bak")
    with open(m2_train, "w", encoding="utf-8") as f:
        f.writelines(new_train)
    with open(m2_val, "w", encoding="utf-8") as f:
        f.writelines(new_val)
    print(f"\nWrote {m2_train} and {m2_val} (backups: *.bak)")


if __name__ == "__main__":
    main()