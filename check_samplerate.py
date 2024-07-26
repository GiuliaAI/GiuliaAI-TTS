import os
import librosa

output_directory = "./GiuliaAI/dataset/GiuliaAI"

for root, _, files in os.walk(output_directory):
    for file in files:
        if file.endswith(".wav"):
            wav_path = os.path.join(root, file)
            y, sr = librosa.load(wav_path, sr=None)
            print(f"{wav_path}: {sr} Hz")
