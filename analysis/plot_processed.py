import numpy as np
import matplotlib.pyplot as plt

file_path = input("Enter processed .npz file path:\n")

data = np.load(file_path)

print("Available keys:", data.files)

signal = data["envelope"]   # IMPORTANT FIX
labels = data["labels"]

t = np.arange(len(signal))

plt.plot(t, signal)
plt.title("Processed EMG (Envelope)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid()

plt.show()