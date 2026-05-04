import os
import numpy as np
import matplotlib.pyplot as plt

# ================= LOAD FILE =================
file_path = input("Enter full path to .npz file:\n")

data = np.load(file_path)

raw = data["data"]          # your EMG signal
labels = data["labels"]     # optional

# ================= TIME AXIS =================
t = np.arange(len(raw))

# ================= PLOT RAW =================
plt.figure()
plt.plot(t, raw)
plt.title("RAW EMG Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

# ================= OPTIONAL: LABEL VIEW =================
plt.figure()
plt.plot(t, raw, label="EMG")

plt.scatter(
    t[labels == 1],
    raw[labels == 1],
    label="Gesture",
    s=5
)

plt.title("EMG with Labels")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.show()