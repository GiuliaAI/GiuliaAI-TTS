import os
import librosa
import numpy as np
from scipy.io import wavfile

def convert_wav_files(input_dir, output_dir, target_sample_rate=22050, target_bit_depth=16):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)

                print(f"Processing {input_path} -> {output_path}")

                y, sr = librosa.load(input_path, sr=None, mono=True)

                if sr != target_sample_rate:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)

                if target_bit_depth == 16:
                    y = (y * 32767).astype(np.int16)

                wavfile.write(output_path, target_sample_rate, y)

                print(f"Saved {output_path}")

input_directory = "./GiuliaAI/dataset/GiuliaAI/origianl"
output_directory = "./GiuliaAI/dataset/GiuliaAI/wav"

convert_wav_files(input_directory, output_directory, target_sample_rate=22050, target_bit_depth=16)
