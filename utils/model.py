import os
import json

import torch
import numpy as np
import argparse

import hifigan
from model import FastSpeech2, ScheduledOptim
from waveglow.model import WaveGlow
from diffwave.model import DiffWave
from diffwave.params import params
from waveglow.arg_parser import parse_waveglow_args
from diffwave.params import AttrDict
from diffwave.inference import predict


def get_model(args, configs, device, train=True):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    vocoder = None
    

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "RUSLAN":
            ckpt = torch.load("hifigan/generator_v1.pth")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "waveglow":
        if speaker == "RUSLAN":
            ckpt = torch.load("waveglow/checkpoint_WaveGlow_550.pt", map_location='cpu')
            #print(ckpt.keys())
            model_config = {
    'n_mel_channels': 80,  # Примерное количество мел-каналов
    'n_flows': 12,  # Примерное количество потоков в WaveGlow
    'n_group': 8,  # Примерное количество групп в WaveGlow
    'n_early_every': 4,  # Примерное значение для параметра early_every
    'n_early_size': 2,  # Примерное значение для параметра early_size
    #'sigma': 1,
    'WN_config':  {
    #'n_mel_channels': 80,
    #'n_in_channels': 4000,
    'n_layers': 8,  # Примерное количество слоев WaveNet в WaveGlow
    'kernel_size': 3,  # Примерный размер ядра свертки в слоях WaveNet
    'n_channels': 512,  # Примерное количество каналов в слоях WaveNet
    },
}
               
            
            vocoder = WaveGlow(**model_config)
            state_dict = ckpt['state_dict']

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # удаляем приставку 'module.'
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value

# Используйте новый словарь состояния для загрузки в модель


            #print(state_dict.keys())
            vocoder.load_state_dict(new_state_dict)
            vocoder.eval()
            vocoder.remove_weightnorm(vocoder)
            vocoder.to(device)

  
    elif name == "diffwave":
        if speaker == "RUSLAN":
            ckpt = torch.load("/content/drive/MyDrive/model_dir/weights-12747.pt", map_location='cpu')
            vocoder = DiffWave(AttrDict(params)).to(device)
            vocoder.load_state_dict(ckpt['model'])
            vocoder.eval()

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    mels = mels.float()
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "waveglow":
            wavs = vocoder.infer(mels)
        elif name == "diffwave":
            wavs, _ = predict(mels, "/content/drive/MyDrive/model_dir/weights-12747.pt", 22050)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
