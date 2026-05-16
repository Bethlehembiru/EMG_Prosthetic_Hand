import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD FILE
# =========================
file_path = input("Enter processed .npz file path:\n")

data = np.load(file_path)

print("Available keys:", data.files)

# =========================
# LOAD SIGNALS
# =========================
filtered = data["filtered"]
rectified = data["rectified"]
envelope = data["envelope"]
labels = data["labels"]

# =========================
# TIME AXIS
# =========================
t = np.arange(len(filtered))

# =========================
# PLOT
# =========================
fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# -------------------------
# FILTERED
# -------------------------
axs[0].plot(t, filtered)
axs[0].set_title("Filtered EMG")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# -------------------------
# RECTIFIED
# -------------------------
axs[1].plot(t, rectified)
axs[1].set_title("Rectified EMG")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)

# -------------------------
# ENVELOPE
# -------------------------
axs[2].plot(t, envelope)
axs[2].set_title("Envelope EMG")
axs[2].set_ylabel("Amplitude")
axs[2].grid(True)

# -------------------------
# LABELS
# -------------------------
axs[3].plot(t, labels)
axs[3].set_title("Labels")
axs[3].set_xlabel("Samples")
axs[3].set_ylabel("Class")
axs[3].grid(True)

plt.tight_layout()
plt.show()