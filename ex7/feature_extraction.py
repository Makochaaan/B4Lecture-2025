import argparse
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def feature_extraction(path_file: list) -> list:
    for path in path_file:
        y,sr = librosa.load(path)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        plt.figure(figsize=(12, 4))
        librosa.display.specshow(mel_spectrogram_db, y_axis='mel', x_axis='time',sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction script")
    parser.add_argument("--input", type=str, required=True, help="Path to the input audio file")
    args = parser.parse_args()
    args.input = args.input.split(",")
    print("Input audio files:", args.input)

    # Example usage of feature_extraction
    features = feature_extraction(args.input)
    print("Extracted features:", features)
