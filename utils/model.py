import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim
from istftnetfe import ISTFTNetFE


def load_pretrained_weights(model, pretrained_path):
    print(f"Loading pretrained weights from {pretrained_path}")
    ckpt = torch.load(pretrained_path)
    pretrained_dict = ckpt["model"]

    model_dict = model.state_dict()
    mismatched_shapes = []

    for name, param in pretrained_dict.items():
        if name in model_dict:
            if model_dict[name].shape != param.shape:
                mismatched_shapes.append((name, model_dict[name].shape, param.shape))
            else:
                model_dict[name] = param

    # Print mismatched shapes
    if mismatched_shapes:
        print("Mismatched shapes found:")
        for name, model_shape, pretrained_shape in mismatched_shapes:
            print(f"{name}: model shape {model_shape}, pretrained shape {pretrained_shape}")

        print("This is not an error, if you know what you're doing.")

    # Load the updated state dict with matching shapes
    model.load_state_dict(model_dict, strict=False)


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = torch.optim.AdamW(model.parameters(),
                                            lr=train_config["optimizer"]["init_lr"],
                                            betas=train_config["optimizer"]["betas"],
                                            eps=train_config["optimizer"]["eps"],
                                            weight_decay=0.001,
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
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)
    elif name == "iSTFTNet":
        vocoder = ISTFTNetFE(None, None)
        vocoder.load_ts("istftnet/universal","cuda")


    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
        elif name == "iSTFTNet":
            wavs = vocoder(mels.float()).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
