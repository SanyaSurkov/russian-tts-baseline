"""Exp 1 baseline evaluation.

Computes per-speaker:
  - MCD (pymcd)
  - F0 RMSE (pyworld)
  - Duration MAE (vs MFA targets)
  - Whisper WER / CER
  - UTMOS (speechmos)
  - Synthesis RTF

Outputs JSON per speaker + aggregate.
"""
import argparse
import json
import os
import time

# NB: these imports require additional installs:
#   pip install pymcd openai-whisper speechmos
# Guarded so this skeleton compiles/imports without eval deps present.
try:
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
    from pymcd.mcd import Calculate_MCD
    import whisper
    from speechmos import utmos
except ImportError:
    torch = None
    np = None
    librosa = None
    sf = None
    Calculate_MCD = None
    whisper = None
    utmos = None
    print("eval deps not installed \u2014 run: pip install pymcd openai-whisper speechmos")


def synthesize_one(model, vocoder, text_ids, speaker_id):
    with torch.no_grad():
        # model.infer signature varies; adapt to repo's synthesize.py
        mel = model(...)  # TODO fill from synthesize.py pattern
        wav = vocoder(mel)
    return wav


def main(args):
    # TODO: load ckpt + vocoder via utils.model.get_model / get_vocoder,
    # iterate test file, synth each utterance, compute metrics, aggregate.
    # body filled after train.py infrastructure is stable
    results = {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out} (skeleton)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--testset", default="scripts/eval_testsets/general_heldout.txt")
    ap.add_argument("--out", default="output/eval/exp1_baseline.json")
    main(ap.parse_args())
