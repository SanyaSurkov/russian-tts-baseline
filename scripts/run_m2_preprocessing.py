# scripts/run_m2_preprocessing.py
"""M2 preprocessing orchestrator: runs accent → post-process TG → validate → preprocessor.

Usage:
    python scripts/run_m2_preprocessing.py \
        --raw_path ./data \
        --m1_tg_path ./preprocessed/MULTISPK_RU/TextGrid \
        --m2_tg_path ./preprocessed/MULTISPK_RU_M2/TextGrid \
        --accent_model turbo \
        --num_workers 8

Stages:
    1. build_stressed_lab over all speakers (reuses existing raw .lab files).
    2. post_process_textgrids (M1 TG → M2 TG).
    2.5 validate_m2_alignment (sampled sanity check).
    3. preprocessor.py with MULTISPK_RU_M2 config (generates mel/pitch/energy/
       duration .npy under preprocessed/MULTISPK_RU_M2/).

Each stage can be skipped with --skip_stage N.
"""
import argparse
import os
import subprocess
import sys


def detect_speakers(raw_path: str) -> list:
    """Auto-detect speaker subdirs in raw_path (each subdir must contain .lab files)."""
    if not os.path.isdir(raw_path):
        raise SystemExit(f"raw_path does not exist: {raw_path}")
    speakers = []
    for d in sorted(os.listdir(raw_path)):
        full = os.path.join(raw_path, d)
        if not os.path.isdir(full):
            continue
        has_lab = any(
            f.endswith(".lab")
            for root, _, files in os.walk(full)
            for f in files
        )
        if has_lab:
            speakers.append(d)
    return speakers


def run(cmd) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    env = os.environ.copy()
    # Корень репозитория (где лежит папка scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", default="./data")
    ap.add_argument("--m1_tg_path", default="./preprocessed/MULTISPK_RU/TextGrid")
    ap.add_argument("--m2_tg_path", default="./preprocessed/MULTISPK_RU_M2/TextGrid")
    ap.add_argument("--preprocess_config", default="config/MULTISPK_RU_M2/preprocess.yaml")
    ap.add_argument("--accent_model", choices=["turbo", "big_poetry", "big"], default="turbo")
    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--skip_stage", type=str, nargs="*", default=[],
                    help='stage ids to skip, e.g. --skip_stage 1 2.5')
    args = ap.parse_args()

    skip = set(args.skip_stage)
    py = sys.executable
    speakers = detect_speakers(args.raw_path)
    print(f"Detected speakers: {speakers}")
    if not speakers:
        raise SystemExit("No speaker directories with .lab files found — check --raw_path")

    if "1" not in skip:
        print("\n=== STAGE 1: build stressed .lab ===")
        run([py, "scripts/build_stressed_lab.py",
             "--raw_path", args.raw_path,
             "--speaker_dirs", *speakers,
             "--accent_model", args.accent_model,
             "--num_workers", str(args.num_workers)])
    else:
        print("Stage 1 skipped")

    if "2" not in skip:
        print("\n=== STAGE 2: post-process TextGrids ===")
        run([py, "scripts/post_process_textgrids.py",
             "--raw_path", args.raw_path,
             "--m1_tg_path", args.m1_tg_path,
             "--m2_tg_path", args.m2_tg_path])
    else:
        print("Stage 2 skipped")

    if "2.5" not in skip:
        print("\n=== STAGE 2.5: validate M2 alignment ===")
        run([py, "scripts/validate_m2_alignment.py",
             "--m1_tg_path", args.m1_tg_path,
             "--m2_tg_path", args.m2_tg_path,
             "--sample_size", "100"])
    else:
        print("Stage 2.5 skipped")

    if "3" not in skip:
        print("\n=== STAGE 3: preprocessor (mel/pitch/energy/duration) ===")
        run([py, "preprocess.py", args.preprocess_config])
    else:
        print("Stage 3 skipped")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
