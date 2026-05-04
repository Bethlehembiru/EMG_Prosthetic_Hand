import os
import time
import numpy as np
import threading
from serial_reader import EMGSerialReader

# ================= SETTINGS =================
REST_TIME = 3
TRANSITION_TIME = 0.5
ACTIVE_TIME = 3

LABELS = {
    "rest": 0,
    "close": 1,
    "open": 2,
    "point": 3,
    "transition": -1
}

# ================= INPUT =================
participant = input("Participant ID: ")
gesture = input("Gesture (close/open/point): ")

# ================= SAVE FOLDER (FIXED PATH) =================
BASE_SAVE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\raw"
save_dir = os.path.join(BASE_SAVE_DIR, f"participant_{participant}")
os.makedirs(save_dir, exist_ok=True)

# ================= SERIAL INIT =================
reader = EMGSerialReader(port='COM3', baudrate=9600)

print("Serial connection established.")

# ================= BASELINE CALIBRATION =================
print("Calibrating baseline... KEEP MUSCLE RELAXED")

time.sleep(2)

cal_samples = []
for _ in range(200):
    val = reader.read_sample()
    if val is not None:
        cal_samples.append(val)

if len(cal_samples) == 0:
    raise Exception("No EMG signal detected during calibration.")

baseline = np.mean(cal_samples)
print(f"Baseline locked at: {baseline:.2f}")

# flush noise
for _ in range(50):
    reader.read_sample()

print("Starting EMG recording...")
print("Press ENTER to stop safely")

# ================= STOP CONTROL =================
running = True

def wait_for_stop():
    input()
    global running
    running = False

threading.Thread(target=wait_for_stop, daemon=True).start()

# ================= STORAGE =================
samples = []
labels = []

start_time = time.time()

cycle_time = REST_TIME + TRANSITION_TIME + ACTIVE_TIME + TRANSITION_TIME

# ================= ACQUISITION LOOP =================
try:
    while running:

        value = reader.read_sample()

        if value is None:
            continue

        # baseline correction
        corrected = value - baseline

        # timing
        t = time.time() - start_time
        phase = t % cycle_time

        # labeling
        if phase < REST_TIME:
            label = LABELS["rest"]

        elif phase < REST_TIME + TRANSITION_TIME:
            label = LABELS["transition"]

        elif phase < REST_TIME + TRANSITION_TIME + ACTIVE_TIME:
            label = LABELS[gesture]

        else:
            label = LABELS["transition"]

        samples.append(corrected)
        labels.append(label)

        print(f"{corrected:.2f} , {label}")

except Exception as e:
    print("Error:", e)

finally:
    print("Saving data...")

    if len(samples) == 0:
        print("WARNING: No samples collected.")
    else:
        samples = np.array(samples, dtype=np.float32)
        labels = np.array(labels, dtype=np.int8)

        filename = os.path.join(
            save_dir,
            f"{gesture}_{int(time.time())}.npz"
        )

        np.savez(filename, data=samples, labels=labels)

        print(f"Saved {len(samples)} samples → {filename}")