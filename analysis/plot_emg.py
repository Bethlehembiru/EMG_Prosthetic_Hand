import numpy as np
import matplotlib.pyplot as plt

file_path = input("Enter .npz file path:\n")

data = np.load(file_path)

raw = data["data"]
labels = data["labels"]

t = np.arange(len(raw))  # SIMPLE AND CORRECT

plt.figure(figsize=(14,6))

plt.plot(t, raw, linewidth=1, label="EMG Signal")

plt.scatter(t[labels == 1], raw[labels == 1], color="black", s=8, label="Close")
plt.scatter(t[labels == 2], raw[labels == 2], color="green", s=8, label="Open")
plt.scatter(t[labels == 3], raw[labels == 3], color="orange", s=8, label="Point")
plt.scatter(t[labels == 4], raw[labels == 4], color="pink", s=8, label="Half Close")
plt.scatter(t[labels == -1], raw[labels == -1], color="red", s=10, marker="x", label="Transition")

plt.title("EMG Signal (Amplitude vs Time)")
plt.xlabel("Samples (time index)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.show()