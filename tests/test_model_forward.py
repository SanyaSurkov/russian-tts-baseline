import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.fastspeech2 import FastSpeech2  # noqa: E402


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "MULTISPK_RU"


def _load_yaml(name):
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_fastspeech2_forward_shapes(fake_preprocessed, monkeypatch):
    # Force CPU even if CUDA is available — module-level `device` globals
    # in utils.tools and model.modules default to cuda when present.
    import utils.tools as _tools
    import model.modules as _modules
    cpu = torch.device("cpu")
    monkeypatch.setattr(_tools, "device", cpu, raising=False)
    monkeypatch.setattr(_modules, "device", cpu, raising=False)

    preprocess_config = _load_yaml("preprocess.yaml")
    model_config = _load_yaml("model.yaml")
    preprocess_config["path"]["preprocessed_path"] = str(fake_preprocessed)

    torch.manual_seed(0)
    model = FastSpeech2(preprocess_config, model_config).cpu()
    model.eval()

    B = 2
    T_src = 6
    T_mel = 20

    speakers = torch.zeros(B, dtype=torch.long)
    texts = torch.randint(low=1, high=100, size=(B, T_src), dtype=torch.long)
    src_lens = torch.tensor([T_src, T_src], dtype=torch.long)
    max_src_len = T_src

    mels = torch.randn(B, T_mel, 80)
    mel_lens = torch.tensor([T_mel, T_mel], dtype=torch.long)
    max_mel_len = T_mel
    p_targets = torch.randn(B, T_src)
    e_targets = torch.randn(B, T_src)

    base = T_mel // T_src
    rem = T_mel - base * T_src
    d_row = [base] * T_src
    for k in range(rem):
        d_row[T_src - 1 - k] += 1
    assert sum(d_row) == T_mel
    d_targets = torch.tensor([d_row, d_row], dtype=torch.long)

    with torch.no_grad():
        out = model(
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
        )

    assert len(out) == 10
    assert out[0].shape == (B, T_mel, 80)
    assert out[1].shape == (B, T_mel, 80)
    assert out[2].shape == (B, T_src)
