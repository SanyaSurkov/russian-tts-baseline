import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config", "MULTISPK_RU")

def _load(name):
    with open(os.path.join(CONFIG_DIR, name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_preprocess_config_keys():
    c = _load("preprocess.yaml")
    assert c["dataset"] == "MULTISPK_RU"
    assert c["preprocessing"]["audio"]["sampling_rate"] == 22050
    assert c["preprocessing"]["pitch"]["feature"] == "phoneme_level"
    assert c["preprocessing"]["energy"]["feature"] == "phoneme_level"
    assert c["path"]["lexicon_path"].endswith("russian_mfa.dict")

def test_model_config_multispeaker():
    c = _load("model.yaml")
    assert c["multi_speaker"] is True
    assert c["transformer"]["encoder_hidden"] == 256
    assert c["transformer"]["decoder_hidden"] == 256

def test_train_config_amp_friendly():
    c = _load("train.yaml")
    assert c["optimizer"]["batch_size"] == 12
    assert c["optimizer"]["grad_acc_step"] == 2
    assert c["step"]["total_step"] == 150000
    assert c["step"]["save_step"] == 10000
