import os
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]

    print(f"Input directory: {in_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Sampling rate: {sampling_rate}")
    print(f"Max wav value: {max_wav_value}")

    print("Processing dataset...")

    with open(os.path.join(in_dir, "transcripts.txt"), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split(":", 1)  # Considera solo il primo ':'
            if len(parts) == 2:
                text, wav_name = parts
                text = text.replace(":", "")  # Rimuovi tutti gli altri ':'
                wav_name += ".wav"
                speaker = "GiuliaAI"
                wav_path = os.path.join(in_dir, speaker, "wav", wav_name)
                print(f"Processing file: {wav_path}")

                if os.path.exists(wav_path):
                    print(f"File exists: {wav_path}")
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sr=sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    out_wav_path = os.path.join(out_dir, speaker, wav_name)
                    wavfile.write(out_wav_path, sampling_rate, wav.astype(np.int16))
                    print(f"Saved processed audio to: {out_wav_path}")
                    lab_path = os.path.join(out_dir, speaker, "{}.lab".format(wav_name[:-4]))
                    with open(lab_path, "w", encoding="utf-8") as f1:
                        f1.write(text)
                    print(f"Saved transcript to: {lab_path}")
                else:
                    print(f"File does not exist: {wav_path}")
            else:
                print(f"Line format error: {line.strip()}")