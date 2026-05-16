import numpy as np
import os
from collections import defaultdict

from SEMG.pre_processing.window import WindowSegmenter
from SEMG.features.extractor import EMGFeatureExtractor


DATA_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\processed"
SAVE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\features"

os.makedirs(SAVE_DIR, exist_ok=True)


segmenter = WindowSegmenter(window_size=250, overlap=0.5)
extractor = EMGFeatureExtractor(fs=1000)


# -----------------------------
# GLOBAL STATISTICS TRACKING
# -----------------------------
global_class_counts = defaultdict(int)
global_rejection_stats = []


for participant_folder in sorted(os.listdir(DATA_DIR)):
    participant_path = os.path.join(DATA_DIR, participant_folder)

    if not os.path.isdir(participant_path):
        continue

    print(f"\n=== Processing {participant_folder} ===")

    X_all = []
    y_all = []

    participant_reject_stats = []

    for file in sorted(os.listdir(participant_path)):
        if not file.endswith(".npz"):
            continue

        path = os.path.join(participant_path, file)
        data = np.load(path)

        signal = data.get("filtered")
        labels = data.get("labels")

        if signal is None or labels is None:
            print(f"Skipping corrupted file: {file}")
            continue

        windows, win_labels, stats = segmenter.segment(signal, labels)

        participant_reject_stats.append(stats)

        if len(windows) == 0:
            continue

        features = extractor.extract(windows)

        if features.shape[0] == 0:
            continue

        X_all.append(features)
        y_all.append(win_labels)

        # -----------------------------
        # update class distribution
        # -----------------------------
        for lbl in win_labels:
            global_class_counts[int(lbl)] += 1

    if len(X_all) == 0:
        print("No valid data for participant")
        continue

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    save_path = os.path.join(SAVE_DIR, f"{participant_folder}_features.npz")

    np.savez(save_path, X=X_all, y=y_all)

    print(f"Saved → {save_path}")
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)

    # store rejection stats
    global_rejection_stats.extend(participant_reject_stats)


# -----------------------------
# FINAL REPORTING (VERY IMPORTANT FOR THESIS)
# -----------------------------
print("\n=== CLASS DISTRIBUTION ===")
for k, v in sorted(global_class_counts.items()):
    print(f"Class {k}: {v} samples")

print("\n=== AVERAGE REJECTION STATS ===")

avg_transition = np.mean([s["transition_rejected"] for s in global_rejection_stats])
avg_mixed = np.mean([s["mixed_label_rejected"] for s in global_rejection_stats])

print(f"Avg transition rejected windows: {avg_transition:.2f}")
print(f"Avg mixed-label rejected windows: {avg_mixed:.2f}")