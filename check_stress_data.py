import os, numpy as np
from collections import defaultdict

BASE = 'C:/Users/Admin/thesis/data/preprocessed_m2'
stats = defaultdict(list)
vowels = ['@a','@a+','@e','@e+','@o','@o+','@i','@i+','@u','@u+']

# Проверяем существование папок
PITCH_DIR = os.path.join(BASE, 'pitch')
ENERGY_DIR = os.path.join(BASE, 'energy')

print(f'Pitch dir exists: {os.path.exists(PITCH_DIR)}')

cnt = 0
errors = 0
with open(os.path.join(BASE, 'train.txt'), encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) < 4: 
            continue
        
        name_id = parts[0]      # Например: 011511_RUSLAN
        speaker = parts[1]      # Например: ruslan_m01
        phones = parts[2].split()
        
        cnt += 1
        
        # Формируем ожидаемое имя файла
        # Паттерн: {speaker}-pitch-{name_id}.npy
        pitch_fname = f'{speaker}-pitch-{name_id}.npy'
        energy_fname = f'{speaker}-energy-{name_id}.npy'
        
        pitch_path = os.path.join(PITCH_DIR, pitch_fname)
        energy_path = os.path.join(ENERGY_DIR, energy_fname)

        if cnt <= 5:
            print(f'Line {cnt}: ID={name_id}, Spk={speaker}')
            print(f'  Looking for: {pitch_fname}')
            print(f'  Exists: {os.path.exists(pitch_path)}')

        try:
            p = np.load(pitch_path)
            e = np.load(energy_path)
        except Exception as ex:
            errors += 1
            if cnt <= 5:
                print(f'  ERROR: {ex}')
            continue
            
        if len(phones) != len(p): 
            if cnt <= 5:
                print(f'  MISMATCH LEN: phones={len(phones)}, pitch={len(p)}')
            continue
            
        for i, ph in enumerate(phones):
            if ph in vowels:
                stats[ph].append((float(p[i]), float(e[i])))
        
        if cnt >= 1000: # Ограничим для теста
            break

print(f'Total lines: {cnt}, Errors: {errors}')
print(f'Stats keys: {list(stats.keys())}')

for ph in ['@a','@a+','@e','@e+','@o','@o+']:
    if ph not in stats: 
        print(f'{ph}: NO DATA')
        continue
    pm = np.mean([x[0] for x in stats[ph]])
    em = np.mean([x[1] for x in stats[ph]])
    print(f'{ph:5}: n={len(stats[ph]):5}, pitch={pm:7.3f}, energy={em:7.3f}')
