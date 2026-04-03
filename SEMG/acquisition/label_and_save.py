import time
import os
import numpy as np
from .fake_serial_reader import EMGSerialReader  # assuming the reader is in acquisition folder

# ===== SETTINGS =====
SAMPLE_RATE = 1000  # Hz

# Timing (seconds)
REST_TIME = 3
TRANSITION_TIME = 0.5  ### around 500 samples are transitions!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ACTIVE_TIME = 3

# Labels
LABELS = {
    "rest": 0,
    "close": 1,
    "open": 2,
    "point": 3,
    "transition": -1
}

# ===== INIT =====
participant = input("Participant ID: ")
gesture = input("Gesture (close/open/point): ")

# Prepare save directory


#save_dir = os.path.join(r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\raw", f"participant_{participant}")
#os.makedirs(save_dir, exist_ok=True)
import os

path = r"C:\Users\hp\PycharmProjects\EMG_Prosthetic_Hand\data\raw"

print("Is DIR:", os.path.isdir(path))
print("Exists:", os.path.exists(path))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

save_dir = os.path.join(BASE_DIR, "data", "raw", f"participant_{participant}")

os.makedirs(save_dir, exist_ok=True)





# Initialize serial reader
reader = EMGSerialReader(port='COM3', baudrate=115200)

samples = []
labels = []

print("Starting recording... Press Ctrl+C to stop.")

start_time = time.time()

try:
    while True:
        value = reader.read_sample()

        if value is None:
            continue  # skip invalid readings

        current_time = time.time() - start_time

        # ===== LABEL LOGIC =====
        cycle_time = REST_TIME + TRANSITION_TIME + ACTIVE_TIME + TRANSITION_TIME
        t = current_time % cycle_time

        if t < REST_TIME:
            label = LABELS["rest"]
        elif t < REST_TIME + TRANSITION_TIME:
            label = LABELS["transition"]
        elif t < REST_TIME + TRANSITION_TIME + ACTIVE_TIME:
            label = LABELS[gesture]
        else:
            label = LABELS["transition"]

        print(f"time: {current_time:.3f}, t: {t:.3f}, label: {label}")

        samples.append(value)
        labels.append(label)

        # Optional: print live stream
        print(f"Sample: {value}, Label: {label}")

except KeyboardInterrupt:
    print("\nRecording stopped by user.")

# ===== SAVE =====
samples = np.array(samples)
labels = np.array(labels)

timestamp = int(time.time())
filename = os.path.join(save_dir, f"{gesture}_{timestamp}.npz")
np.savez(filename, data=samples, labels=labels)

print(f"Saved {len(samples)} samples to {filename}")