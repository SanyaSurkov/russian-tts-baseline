# scripts/diagnose_vowel_labels.py
"""One-off: scan all M1 TextGrids, print unique phone labels that look like vowels.

Usage:
    python scripts/diagnose_vowel_labels.py

Output: printed sorted list of unique labels in phone-tier that contain at least one
IPA vowel character. Use the output to decide which stressed variants to add to
symbols.py (Task 7).
"""
import os
import sys
from collections import Counter
import tgt

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

M1_TG_DIR = "C:/Users/Admin/thesis/data/preprocessed/TextGrid"
VOWEL_CHARS = set("aeiouɨəʉɔɛɪʊyø")


def main():
    label_counts = Counter()
    n_files = 0
    for root, _, files in os.walk(M1_TG_DIR):
        for f in files:
            if not f.endswith(".TextGrid"):
                continue
            path = os.path.join(root, f)
            try:
                tg = tgt.io.read_textgrid(path)
            except Exception as e:
                print(f"  skip {path}: {e}")
                continue
            n_files += 1
            try:
                phone_tier = tg.get_tier_by_name("phones")
            except Exception:
                continue
            for interval in phone_tier.intervals:
                label = interval.text.strip()
                if label and any(c in VOWEL_CHARS for c in label):
                    label_counts[label] += 1

    print(f"Scanned {n_files} TextGrid files")
    print(f"Unique vowel-like labels: {len(label_counts)}")
    print("\nlabel\tcount")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"{label}\t{count}")


if __name__ == "__main__":
    main()
