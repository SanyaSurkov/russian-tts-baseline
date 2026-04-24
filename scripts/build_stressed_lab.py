# scripts/build_stressed_lab.py
"""Apply ruaccent to every .lab in data/, save as .lab.stressed sibling.

Usage:
    python scripts/build_stressed_lab.py \
        --raw_path ./data \
        --speaker_dirs ruslan mailabs_m01 mailabs_m02 mailabs_m03 \
        --accent_model turbo \
        --num_workers 1

Output: for every <path>/<basename>.lab creates <path>/<basename>.lab.stressed
with ruaccent-marked text on a single line.

Workers init their own RUAccent instance once (via initializer). With 6-8
workers on a modern CPU, wall-clock drops ~6x on 40k-file datasets.
"""
import argparse
import multiprocessing as mp
import os
from tqdm import tqdm


_worker_accentizer = None


def _worker_init(model_size: str) -> None:
    """One-time per-worker RUAccent load."""
    global _worker_accentizer
    from ruaccent import RUAccent
    a = RUAccent()
    a.load(omograph_model_size=model_size, use_dictionary=True)
    _worker_accentizer = a


def _worker_stress_one(path: str) -> int:
    """Stress a single .lab file. Returns 1 on success, 0 on error."""
    import re
    _RU_VOWELS = "аеёиоуыэюяaeiou"
    _VOWEL_RE = re.compile(f"[{_RU_VOWELS}]", re.IGNORECASE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.readline().strip()
        stressed = _worker_accentizer.process_all(text)

        # Log words ruaccent couldn't stress (no heuristic fallback).
        tokens = re.split(r"(\W+)", stressed)
        unstressed = [t for t in tokens if t and _VOWEL_RE.search(t) and "+" not in t]
        if unstressed:
            os.makedirs("logs", exist_ok=True)
            with open("logs/unstressed_words.txt", "a", encoding="utf-8") as lf:
                for w in unstressed:
                    lf.write(w + "\n")

        with open(path + ".stressed", "w", encoding="utf-8") as f:
            f.write(stressed + "\n")
        return 1
    except Exception as e:
        print(f"ERR {path}: {e}")
        return 0


def iter_lab_files(raw_path, speakers):
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
    ap.add_argument("--accent_model", choices=["turbo", "big_poetry", "big"], default="turbo",
                    help="ruaccent omograph model size; 'big' is most accurate but ~4x slower")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="parallel workers (each loads its own RUAccent; 6-8 is a good default on modern CPU)")
    args = ap.parse_args()

    lab_paths = list(iter_lab_files(args.raw_path, args.speaker_dirs))
    print(f"Found {len(lab_paths)} .lab files; workers={args.num_workers}; model={args.accent_model}")

    if args.num_workers <= 1:
        _worker_init(args.accent_model)
        n_ok = sum(_worker_stress_one(p) for p in tqdm(lab_paths))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=(args.accent_model,),
        ) as pool:
            n_ok = sum(tqdm(
                pool.imap_unordered(_worker_stress_one, lab_paths, chunksize=32),
                total=len(lab_paths),
            ))

    print(f"Wrote {n_ok}/{len(lab_paths)} stressed .lab files")
    if os.path.exists("logs/unstressed_words.txt"):
        with open("logs/unstressed_words.txt", encoding="utf-8") as f:
            n_unstressed = sum(1 for _ in f)
        print(f"Words left unstressed (ruaccent had no stress): {n_unstressed} (see logs/unstressed_words.txt)")


if __name__ == "__main__":
    main()
