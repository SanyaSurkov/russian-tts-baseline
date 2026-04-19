import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import Dataset  # noqa: E402
from text import text_to_sequence  # noqa: E402


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "MULTISPK_RU"


def _load_yaml(name):
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_dataset_collate_shapes(fake_preprocessed):
    rng = np.random.default_rng(seed=0)
    d = fake_preprocessed

    for sub in ("mel", "pitch", "energy", "duration"):
        (d / sub).mkdir(exist_ok=True)

    cleaners = ["basic_cleaners"]
    text = "п р и в е т"
    raw = "привет"

    utterances = []
    for i in range(4):
        basename = f"u{i}"
        speaker = "spk0" if i % 2 == 0 else "spk1"

        phone = np.array(text_to_sequence(text, cleaners))
        t_src = int(phone.shape[0])
        t_mel = 20 + i * 5

        mel = rng.standard_normal((t_mel, 80)).astype(np.float32)
        pitch = rng.standard_normal((t_src,)).astype(np.float32)
        energy = rng.standard_normal((t_src,)).astype(np.float32)

        duration = np.ones((t_src,), dtype=np.int32)
        remainder = t_mel - int(duration.sum())
        duration[-1] += remainder
        assert int(duration.sum()) == t_mel

        np.save(d / "mel" / f"{speaker}-mel-{basename}.npy", mel)
        np.save(d / "pitch" / f"{speaker}-pitch-{basename}.npy", pitch)
        np.save(d / "energy" / f"{speaker}-energy-{basename}.npy", energy)
        np.save(d / "duration" / f"{speaker}-duration-{basename}.npy", duration)

        utterances.append((basename, speaker, text, raw))

    with open(d / "train.txt", "w", encoding="utf-8") as f:
        for basename, speaker, t, r in utterances:
            f.write(f"{basename}|{speaker}|{t}|{r}\n")

    preprocess_config = _load_yaml("preprocess.yaml")
    train_config = _load_yaml("train.yaml")
    preprocess_config["path"]["preprocessed_path"] = str(d)
    train_config["optimizer"]["batch_size"] = 2

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    assert len(dataset) == 4

    samples = [dataset[i] for i in range(4)]
    batches = dataset.collate_fn(samples)

    assert isinstance(batches, list)
    assert len(batches) >= 1

    batch = batches[0]
    assert len(batch) == 12
    assert len(batch[0]) == 2
    assert batch[6].shape[0] == 2 and batch[6].shape[2] == 80
    assert batch[3].shape[0] == 2
