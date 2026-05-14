import os
import time
import numpy as np
from serial_reader import EMGSerialReader

# ================= SETTINGS =================

REST_TIME = 3
ACTIVE_TIME = 3
TRANSITION_TIME = 0.5
COUNTDOWN_TIME = 3
CYCLES = 10

LABELS = {
    "rest": 0,
    "close": 1,
    "half_close": 4,
    "open": 2,
    "point": 3,
    "transition": -1
}

# ================= USER INPUT =================

participant = input("Participant ID: ")
gesture = input("Gesture (close/open/point/half_close): ").strip().lower()

if gesture not in LABELS:
    raise ValueError("Invalid gesture")

# ================= SAVE DIRECTORY =================

save_dir = os.path.join(
    r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\raw",
    f"participant_{participant}"
)

os.makedirs(save_dir, exist_ok=True)

# ================= SERIAL CONNECTION =================

reader = EMGSerialReader(port='COM3', baudrate=9600)

# ================= BASELINE CALIBRATION =================

print("\nCalibrating baseline...")
time.sleep(2)

cal = []

for _ in range(300):

    v = reader.read_sample()

    if v is not None:
        cal.append(v)

baseline = np.mean(cal)

print(f"Baseline: {baseline:.2f}")

# ================= DATA STORAGE =================

samples = []
labels = []

# ================= FUNCTIONS =================

def clear_buffer():
    """
    Remove old serial data gathered
    during countdown.
    """
    reader.ser.reset_input_buffer()

def countdown(msg, cycle_num):

    print(f"\n===== CYCLE {cycle_num}/{CYCLES} =====")
    print(msg)

    for i in range(COUNTDOWN_TIME, 0, -1):

        print(i)
        time.sleep(1)

    # Remove stale samples
    clear_buffer()

def collect(duration, label, name):

    print(f"\nCollecting: {name}")

    start = time.time()

    while time.time() - start < duration:

        v = reader.read_sample()

        if v is None:
            continue

        processed = v - baseline

        samples.append(processed)
        labels.append(label)

        print(f"{processed:.2f}, {name}")

def transition():

    collect(
        TRANSITION_TIME,
        LABELS["transition"],
        "TRANSITION"
    )

# ================= MAIN LOOP =================

for c in range(CYCLES):

    cycle_num = c + 1

    # ---------- REST ----------

    countdown("Prepare REST", cycle_num)

    # transition AFTER countdown but BEFORE rest
    transition()

    collect(
        REST_TIME,
        LABELS["rest"],
        "REST"
    )

    # ---------- ACTIVE GESTURE ----------

    countdown(
        f"Prepare {gesture.upper()}",
        cycle_num
    )

    # transition AFTER countdown but BEFORE contraction
    transition()

    collect(
        ACTIVE_TIME,
        LABELS[gesture],
        gesture.upper()
    )

# ================= SAVE DATA =================

samples = np.array(samples, dtype=np.float32)
labels = np.array(labels, dtype=np.int8)

file = os.path.join(
    save_dir,
    f"{gesture}_{int(time.time())}.npz"
)

np.savez(file, data=samples, labels=labels)

print("\nSaved:", file)