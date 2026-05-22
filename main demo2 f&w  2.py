import numpy as np
import os
from collections import defaultdict

from SEMG.pre_processing.window import WindowSegmenter
from SEMG.features.enveloped_extractor import EnvelopeFeatureExtractor


# =====================================================
# PATHS
# =====================================================
DATA_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\processed_close"
SAVE_DIR = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\features_close\enveloped"

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# WINDOWING + FEATURE EXTRACTION
# =====================================================
segmenter = WindowSegmenter(
    window_size=250,
    overlap=0.5
)

extractor = EnvelopeFeatureExtractor(fs=1000)


# =====================================================
# GLOBAL STATISTICS
# =====================================================
global_class_counts = defaultdict(int)
global_rejection_stats = []


# =====================================================
# PROCESS PARTICIPANTS
# =====================================================
for participant_folder in sorted(os.listdir(DATA_DIR)):

    participant_path = os.path.join(DATA_DIR, participant_folder)

    if not os.path.isdir(participant_path):
        continue

    print(f"\n=== Processing {participant_folder} ===")

    X_all = []
    y_all = []
    participant_reject_stats = []

    # =================================================
    # PROCESS FILES
    # =================================================
    for file in sorted(os.listdir(participant_path)):

        if not file.endswith(".npz"):
            continue

        path = os.path.join(participant_path, file)
        data = np.load(path)

        # =================================================
        # LOAD SIGNALS (ALL PRECOMPUTED)
        # =================================================
        filtered = data.get("filtered")
        rectified = data.get("rectified")
        envelope = data.get("envelope")
        labels = data.get("labels")

        # =================================================
        # SAFETY CHECK
        # =================================================
        if filtered is None or rectified is None or labels is None:
            print(f"Skipping corrupted file: {file}")
            continue

        # =================================================
        # WINDOWING
        # =================================================
        filtered_windows, win_labels, stats = segmenter.segment(
            filtered,
            labels
        )

        rectified_windows, _, _ = segmenter.segment(
            rectified,
            labels
        )

        # envelope fallback (safe handling)
        if envelope is not None:
            envelope_windows, _, _ = segmenter.segment(
                envelope,
                labels
            )
        else:
            envelope_windows = np.mean(
                rectified_windows,
                axis=1,
                keepdims=True
            )

        participant_reject_stats.append(stats)

        # =================================================
        # VALIDITY CHECK
        # =================================================
        if len(filtered_windows) == 0:
            continue

        if not (
            len(filtered_windows)
            == len(rectified_windows)
            == len(envelope_windows)
            == len(win_labels)
        ):
            print(f"Window mismatch detected in {file}")
            continue

        # =================================================
        # FEATURE EXTRACTION
        # =================================================
        features = extractor.extract(
            envelope_windows)

        if features.shape[0] == 0:
            continue

        # =================================================
        # STORE
        # =================================================
        X_all.append(features)
        y_all.append(win_labels)

        for lbl in win_labels:
            global_class_counts[int(lbl)] += 1

    # =====================================================
    # PARTICIPANT VALIDATION
    # =====================================================
    if len(X_all) == 0:
        print("No valid data for participant")
        continue

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    # =====================================================
    # SAVE
    # =====================================================
    save_path = os.path.join(
        SAVE_DIR,
        f"{participant_folder}_features.npz"
    )

    np.savez(save_path, X=X_all, y=y_all)

    print(f"Saved → {save_path}")
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)

    global_rejection_stats.extend(participant_reject_stats)


# =====================================================
# FINAL REPORTING
# =====================================================
print("\n=== CLASS DISTRIBUTION ===")

for k, v in sorted(global_class_counts.items()):
    print(f"Class {k}: {v} samples")


print("\n=== AVERAGE REJECTION STATS ===")

avg_transition = np.mean([s["transition_rejected"] for s in global_rejection_stats])
avg_mixed = np.mean([s["mixed_label_rejected"] for s in global_rejection_stats])

print(f"Avg transition rejected windows: {avg_transition:.2f}")
print(f"Avg mixed-label rejected windows: {avg_mixed:.2f}")

print("\nFeature extraction completed successfully.")