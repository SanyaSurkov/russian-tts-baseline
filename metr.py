# -*- coding: utf-8 -*-
"""ЛОКАЛЬНЫЙ запуск — файлы берутся напрямую с диска ПК"""


import torch
import torchaudio
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as nisqa
import os
import glob
import numpy as np

def load_audio_for_nisqa(file_path, target_sr=16000):
    waveform, orig_sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0).float()
    return waveform, target_sr

def predict_mos_nisqa(audio_path):
    waveform, sr = load_audio_for_nisqa(audio_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    waveform = waveform.to(device)
    with torch.no_grad():
        mos, noisiness, discontinuity, coloration, loudness = nisqa(waveform, sr)
    return {
        "MOS": mos.item(),
        "Noisiness": noisiness.item(),
        "Discontinuity": discontinuity.item(),
        "Coloration": coloration.item(),
        "Loudness": loudness.item()
    }

# ═══════════════════════════════════════════════════════════════
# ПУТЬ К ПАПКЕ НА ВАШЕМ ПК (Windows)
# ═══════════════════════════════════════════════════════════════
AUDIO_FOLDER = r"C:\Users\Admin\thesis\russian-tts-baseline\output\result\MULTISPK_RU_GAVNO"

extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")
audio_files = []
for ext in extensions:
    audio_files.extend(glob.glob(os.path.join(AUDIO_FOLDER, ext)))
audio_files = sorted(audio_files)[:10]

print(f"🔍 Найдено файлов: {len(audio_files)}")

all_scores = {"MOS": [], "Noisiness": [], "Discontinuity": [], "Coloration": [], "Loudness": []}

for i, audio_file in enumerate(audio_files, 1):
    filename = os.path.basename(audio_file)
    print(f"\n[{i}/{len(audio_files)}] 🎧 {filename}")
    try:
        scores = predict_mos_nisqa(audio_file)
        print(f"   MOS: {scores['MOS']:.3f} | Шум: {scores['Noisiness']:.3f} | "
              f"Прерыв: {scores['Discontinuity']:.3f} | Тембр: {scores['Coloration']:.3f} | "
              f"Громк: {scores['Loudness']:.3f}")
        for key in all_scores:
            all_scores[key].append(scores[key])
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")

# СВОДКА
print("\n" + "=" * 65)
print("📊 СРЕДНИЕ МЕТРИКИ ПО ВСЕМ ФАЙЛАМ")
print("=" * 65)
for metric_name, values in all_scores.items():
    if values:
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        labels = {
            "MOS": "MOS (общий)        ",
            "Noisiness": "Шумность           ",
            "Discontinuity": "Прерывистость      ",
            "Coloration": "Окраска (тембр)    ",
            "Loudness": "Громкость          "
        }
        print(f"   {labels[metric_name]}: {mean_val:.3f}  (σ={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f})")
print("=" * 65)
print("📌 Шкала: 1 = плохо, 5 = отлично")