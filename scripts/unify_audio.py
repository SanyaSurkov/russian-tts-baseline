"""Resample all source audio to 22050 Hz 16-bit mono WAV, organized per speaker.

Output layout:
  data/unified/<speaker_id>/<basename>.wav
  data/unified/<speaker_id>/<basename>.lab   # transcript (one utterance per file)
"""
import argparse
import os
import soundfile as sf
import librosa
from tqdm import tqdm


def resample_write(src_path, dst_path, target_sr=22050):
    audio, sr = librosa.load(src_path, sr=target_sr, mono=True)
    sf.write(dst_path, audio, target_sr, subtype='PCM_16')


def process_ruslan(src_root, dst_root):
    wavs_dir = os.path.join(src_root, "wavs")
    meta = os.path.join(src_root, "metadata.csv")
    if not os.path.isfile(meta):
        print(f"[ruslan] skip: {meta} not found")
        return
    spk = "ruslan_m01"
    out_dir = os.path.join(dst_root, spk)
    os.makedirs(out_dir, exist_ok=True)
    with open(meta, encoding="utf-8") as f:
        lines = list(f)
    for line in tqdm(lines, desc=spk):
        parts = line.strip().split("|")
        if len(parts) < 2: continue
        base, text = parts[0], parts[1]
        src = os.path.join(wavs_dir, f"{base}.wav")
        if not os.path.isfile(src): continue
        resample_write(src, os.path.join(out_dir, f"{base}.wav"))
        with open(os.path.join(out_dir, f"{base}.lab"), "w", encoding="utf-8") as g:
            g.write(text)


def process_mailabs(src_root, dst_root):
    by_book = os.path.join(src_root, "ru_RU", "by_book")
    if not os.path.isdir(by_book):
        print(f"[m-ailabs] skip: {by_book} not found")
        return
    for gender in os.listdir(by_book):
        gdir = os.path.join(by_book, gender)
        if not os.path.isdir(gdir): continue
        for i, spk in enumerate(sorted(os.listdir(gdir))):
            spk_id = f"mailabs_{gender[0]}{i+1:02d}"
            out_dir = os.path.join(dst_root, spk_id)
            os.makedirs(out_dir, exist_ok=True)
            for book in os.listdir(os.path.join(gdir, spk)):
                book_dir = os.path.join(gdir, spk, book)
                meta = os.path.join(book_dir, "metadata.csv")
                wavs = os.path.join(book_dir, "wavs")
                if not os.path.isfile(meta): continue
                with open(meta, encoding="utf-8") as f:
                    lines = list(f)
                for line in tqdm(lines, desc=f"{spk_id}/{book}"):
                    parts = line.strip().split("|")
                    if len(parts) < 2: continue
                    base, text = parts[0], parts[1]
                    src = os.path.join(wavs, f"{base}.wav")
                    if not os.path.isfile(src): continue
                    dst_base = f"{book}_{base}"
                    resample_write(src, os.path.join(out_dir, f"{dst_base}.wav"))
                    with open(os.path.join(out_dir, f"{dst_base}.lab"), "w", encoding="utf-8") as g:
                        g.write(text)


def process_sova(src_root, dst_root, keep_speakers):
    if not keep_speakers or not os.path.isdir(src_root):
        return
    for spk in os.listdir(src_root):
        if spk not in keep_speakers: continue
        sdir = os.path.join(src_root, spk)
        meta = os.path.join(sdir, "metadata.csv")
        if not os.path.isfile(meta): continue
        spk_id = keep_speakers[spk]
        out_dir = os.path.join(dst_root, spk_id)
        os.makedirs(out_dir, exist_ok=True)
        with open(meta, encoding="utf-8") as f:
            lines = list(f)
        for line in tqdm(lines, desc=spk_id):
            parts = line.strip().split("|")
            if len(parts) < 2: continue
            base, text = parts[0], parts[1]
            src_opus = os.path.join(sdir, f"{base}.opus")
            if not os.path.isfile(src_opus): continue
            resample_write(src_opus, os.path.join(out_dir, f"{base}.wav"))
            with open(os.path.join(out_dir, f"{base}.lab"), "w", encoding="utf-8") as g:
                g.write(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="C:/Users/Admin/thesis/data/raw")
    ap.add_argument("--dst_root", default="C:/Users/Admin/thesis/data/unified")
    args = ap.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    process_ruslan(os.path.join(args.raw_root, "ruslan"), args.dst_root)
    process_mailabs(os.path.join(args.raw_root, "m-ailabs"), args.dst_root)

    # SOVA disabled by default — M-AILABS covers female voice for M1.
    # To re-enable: fill this dict and remove the `if sova_keep:` guard.
    sova_keep = {}
    if sova_keep:
        process_sova(os.path.join(args.raw_root, "sova"), args.dst_root, sova_keep)
