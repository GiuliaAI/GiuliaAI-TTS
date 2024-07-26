import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss, FastSpeech3Loss
from dataset import Dataset
import numpy as np
from model import FastSpeech2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_FastSpeech2(chkpname, pre_config_path, model_config_path):
    checkpoint_path = chkpname

    p_config = yaml.load(open(pre_config_path, "r"), Loader=yaml.FullLoader)
    m_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)

    fmodel = FastSpeech2(p_config, m_config)
    fmodel.load_state_dict(torch.load(checkpoint_path)['model'])
    fmodel.requires_grad = False
    fmodel.eval()
    fmodel.to(device)
    return fmodel


def extract_postnets(model, step, configs, args):
    preprocess_config, model_config, train_config = configs

    out_dir = args.out_path
    train_config["optimizer"]["batch_size"] = 8

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Get dataset
    dataset = Dataset(
        args.source, preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function

    n_extracted = 0
    # Evaluation
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                postnets = output[1]
                mel_lens = output[9]

                ids = batch[0]

                postnets = postnets.cpu().numpy()
                mel_lens = mel_lens.cpu().numpy()

                for i in range(0, postnets.shape[0]):
                    # vocoder takes in (batch, channels, len); this is (batch, len, channels)
                    postnet_unpadded = np.transpose(postnets[i, :mel_lens[i]])
                    out_fn = f"{ids[i]}.npy"
                    out_full_fn = os.path.join(out_dir, out_fn)
                    np.save(out_full_fn, postnet_unpadded)
                    n_extracted += 1

    return n_extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="path out for postnets",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="eval or train",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_FastSpeech2(args.checkpoint,
                            args.preprocess_config,
                            args.model_config)

    extracted = extract_postnets(model, 0, configs, args)
    print(f"Extracted {extracted} postnets.")
