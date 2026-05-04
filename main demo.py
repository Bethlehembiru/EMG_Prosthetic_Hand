import numpy as np
import os
from SEMG.pre_processing.filters import EMGFilter

base_dir = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\raw"
save_dir = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\processed"

emg_filter = EMGFilter(fs=1000)

for participant_folder in os.listdir(base_dir):
    participant_path = os.path.join(base_dir, participant_folder)

    if not os.path.isdir(participant_path):
        continue

    participant_id = participant_folder.split("_")[-1]

    for file in os.listdir(participant_path):
        if not file.endswith(".npz"):
            continue

        file_path = os.path.join(participant_path, file)

        print(f"Processing: {file_path}")

        data = np.load(file_path)

        signal = data["data"]
        labels = data["labels"]

        gesture = file.split("_")[0]

        # processing here
        envelope = emg_filter.apply(signal)

        # Saving happens HERE (controlled)
        save_path = os.path.join(
            save_dir,
            f"participant_{participant_id}"
        )

        os.makedirs(save_path, exist_ok=True)

        filename = os.path.join(
            save_path,
            f"{gesture}_{np.random.randint(1e6)}.npz"
        )

        np.savez(filename, envelope=envelope, labels=labels)

print("All participants processed.")