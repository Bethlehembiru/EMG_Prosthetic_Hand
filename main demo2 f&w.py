from SEMG.pre_processing.window import WindowSegmenter
from SEMG.features.extractor import EMGFeatureExtractor
import numpy as np
import os

DATA_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\processed"
SAVE_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\features"

os.makedirs(SAVE_DIR, exist_ok=True)

segmenter = WindowSegmenter(window_size=250, overlap=0.5)
extractor = EMGFeatureExtractor(fs=1000)

for participant_folder in os.listdir(DATA_DIR):
    participant_path = os.path.join(DATA_DIR, participant_folder)

    if not os.path.isdir(participant_path):
        continue

    print("Entering:", participant_folder)

    for file in os.listdir(participant_path):
        if not file.endswith(".npz"):
            continue

        path = os.path.join(participant_path, file)
        print("Processing file:", path)

        data = np.load(path)

        signal = data["envelope"]
        labels = data["labels"]

        windows, win_labels = segmenter.segment(signal, labels)

        print("Windows shape:", windows.shape)

        if len(windows) == 0:
            print("No valid windows → skipped")
            continue

        features = extractor.extract(windows)

        gesture = file.split("_")[0]

        save_path = os.path.join(
            SAVE_DIR,
            f"{participant_folder}_{gesture}_{np.random.randint(1e6)}.npz"
        )

        np.savez(save_path, X=features, y=win_labels)

        print("Saved →", save_path)