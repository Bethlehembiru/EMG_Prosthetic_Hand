import numpy as np
import os

# ===== SETTINGS =====
DATA_DIR = "../../data/raw"         # folder with raw labeled .npz files
PROCESSED_DIR = "../../data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE = 250                    # adjust to 250-300 as needed
WINDOW_STEP = WINDOW_SIZE // 2       # 50% overlap
REJECT_LABEL = -1                    # transition label

# ===== FUNCTION =====
def segment_and_save(file_path, participant_name):
    data = np.load(file_path)
    samples = data["data"]
    labels = data["labels"]

    windows = []
    window_labels = []

    for start in range(0, len(samples) - WINDOW_SIZE + 1, WINDOW_STEP):
        end = start + WINDOW_SIZE
        window = samples[start:end]
        window_label = labels[start:end]

        # Reject window if any transition label
        if REJECT_LABEL in window_label:
            continue

        # Assign window label as the most frequent
        unique, counts = np.unique(window_label, return_counts=True)
        final_label = unique[np.argmax(counts)]

        windows.append(window)
        window_labels.append(final_label)

    if not windows:
        print(f"No valid windows in {file_path}")
        return

    windows = np.array(windows)
    window_labels = np.array(window_labels)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(PROCESSED_DIR, f"{participant_name}_{base_name}_windows.npz")
    np.savez(save_path, windows=windows, labels=window_labels)
    print(f"Saved segmented windows to {save_path}")

# ===== MAIN LOOP =====
for participant_folder in os.listdir(DATA_DIR):
    participant_path = os.path.join(DATA_DIR, participant_folder)
    if not os.path.isdir(participant_path):
        continue

    for file in os.listdir(participant_path):
        if file.endswith(".npz"):
            file_path = os.path.join(participant_path, file)
            segment_and_save(file_path, participant_folder)