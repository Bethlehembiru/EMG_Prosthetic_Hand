import numpy as np
import os

npz_path = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\raw\participant_1\close_1775228373.npz"

# get the folder where the npz file lives
folder = os.path.dirname(npz_path)

# create csv path in the same folder
csv_path = os.path.join(folder, "output.csv")

data = np.load(npz_path)

print("Keys:", data.files)

signals = data["data"]
labels = data["labels"]

combined = np.column_stack((signals, labels))

np.savetxt(csv_path, combined, delimiter=",", header="signal,label", comments="")

print("Saved to:", csv_path)