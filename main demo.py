import numpy as np
import os

from SEMG import pre-processing # change this to your actual file name

# Base directory containing all participants
base_dir = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\raw"

# Initialize filter
emg_filter = EMGFilter(fs=1000)

# Loop through participants
for participant_folder in os.listdir(base_dir):
    participant_path = os.path.join(base_dir, participant_folder)

    if not os.path.isdir(participant_path):
        continue

    # Extract participant ID (e.g., "participant_1" → "1")
    participant_id = participant_folder.split("_")[-1]

    # Loop through each file (each gesture recording)
    for file in os.listdir(participant_path):
        if not file.endswith(".npz"):
            continue

        file_path = os.path.join(participant_path, file)

        print(f"Processing: {file_path}")

        # Load data
        data = np.load(file_path)

        signal = data["data"]
        labels = data["labels"]

        # Extract gesture from filename (close_123.npz → "close")
        gesture = file.split("_")[0]

        # Apply filtering + save processed version
        emg_filter.apply(
            signal,
            labels=labels,
            participant=participant_id,
            gesture=gesture
        )

print("All participants processed.")