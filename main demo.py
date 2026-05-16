import numpy as np
import os

from SEMG.pre_processing.filters import EMGFilter

# =========================
# PATHS
# =========================
BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\raw"
SAVE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\processed"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# FILTER
# =========================
emg_filter = EMGFilter(fs=1000)

# =========================
# PROCESS PARTICIPANTS
# =========================
for participant_folder in os.listdir(BASE_DIR):

    participant_path = os.path.join(BASE_DIR, participant_folder)

    if not os.path.isdir(participant_path):
        continue

    participant_id = participant_folder.split("_")[-1]

    print(f"\n=== Processing Participant {participant_id} ===")

    for file in os.listdir(participant_path):

        if not file.endswith(".npz"):
            continue

        file_path = os.path.join(participant_path, file)

        print(f"Processing: {file}")

        data = np.load(file_path)

        signal = data["data"]
        labels = data["labels"]

        gesture = file.split("_")[0]

        # =========================
        # FILTERING
        # =========================
        filtered, rectified, envelope = emg_filter.apply(
            signal,
            return_signals=("filtered", "rectified", "envelope")
        )

        # =========================
        # SAVE DIRECTORY
        # =========================
        participant_save_path = os.path.join(
            SAVE_DIR,
            f"participant_{participant_id}"
        )

        os.makedirs(participant_save_path, exist_ok=True)

        # =========================
        # SAVE FILE
        # =========================
        save_file = os.path.join(
            participant_save_path,
            f"{gesture}_{np.random.randint(1e6)}.npz"
        )

        np.savez(
            save_file,

            filtered=filtered,
            rectified=rectified,
            envelope=envelope,

            labels=labels
        )

        print(f"Saved → {save_file}")

print("\nAll participants processed successfully.")