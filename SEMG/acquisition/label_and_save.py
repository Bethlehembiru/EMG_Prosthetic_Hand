import os
import time
import numpy as np
from serial_reader import EMGSerialReader

# ================= SETTINGS =================
REST_TIME = 3
TRANSITION_TIME = 0.5
ACTIVE_TIME = 3
COUNTDOWN_TIME = 3
CYCLES = 20

LABELS = {
    "rest": 0,
    "close": 1,
    "open": 2,
    "point": 3,
    "transition": -1
}

# ================= INPUT =================
participant = input("Participant ID: ")
gesture = input("Gesture (close/open/point): ").strip().lower()

if gesture not in ["close", "open", "point"]:
    raise ValueError("Invalid gesture.")

# ================= SAVE DIRECTORY =================
BASE_SAVE_DIR = r"C:\Users\HP\OneDrive\Desktop\EMG_Prosthetic_Hand\data\raw"

save_dir = os.path.join(BASE_SAVE_DIR, f"participant_{participant}")
os.makedirs(save_dir, exist_ok=True)

# ================= SERIAL INIT =================
reader = EMGSerialReader(port='COM3', baudrate=9600)

print("\nSerial connection established.")

# ================= BASELINE CALIBRATION =================
print("\n===================================")
print("BASELINE CALIBRATION")
print("KEEP MUSCLE RELAXED")
print("===================================")

time.sleep(2)

cal_samples = []

for _ in range(200):

    val = reader.read_sample()

    if val is not None:
        cal_samples.append(val)

if len(cal_samples) == 0:
    raise Exception("No EMG signal detected during calibration.")

baseline = np.mean(cal_samples)

print(f"\nBaseline locked at: {baseline:.2f}")

# Flush noise
for _ in range(50):
    reader.read_sample()

# ================= STORAGE =================
samples = []
labels = []

# ================= HELPER FUNCTIONS =================
def countdown(message):

    print(f"\n{message}")

    for i in range(COUNTDOWN_TIME, 0, -1):
        print(i)
        time.sleep(1)


def collect_phase(duration, label, phase_name):

    start = time.time()

    while time.time() - start < duration:

        val = reader.read_sample()

        if val is None:
            continue

        corrected = val - baseline

        samples.append(corrected)
        labels.append(label)

        print(f"{corrected:.2f} , {phase_name}")


# ================= START =================
print("\n===================================")
print("GUIDED EMG ACQUISITION STARTED")
print(f"Gesture : {gesture.upper()}")
print(f"Cycles  : {CYCLES}")
print("===================================")

try:

    for cycle in range(CYCLES):

        current_cycle = cycle + 1

        print("\n===================================")
        print(f"CYCLE {current_cycle} / {CYCLES}")
        print("===================================")

        # ==================================================
        # PREPARE FOR REST
        # ==================================================
        countdown(
            f"Prepare to REST\n"
            f"Upcoming Phase: REST\n"
            f"Cycle {current_cycle}/{CYCLES}"
        )

        print("\nREST")

        collect_phase(
            REST_TIME,
            LABELS["rest"],
            "REST"
        )

        # ==================================================
        # TRANSITION TO ACTIVE
        # ==================================================
        print("\nTRANSITION")

        collect_phase(
            TRANSITION_TIME,
            LABELS["transition"],
            "TRANSITION"
        )

        # ==================================================
        # PREPARE FOR CONTRACTION
        # ==================================================
        countdown(
            f"Prepare to CONTRACT\n"
            f"Gesture: {gesture.upper()}\n"
            f"Cycle {current_cycle}/{CYCLES}"
        )

        print(f"\nCONTRACT -> {gesture.upper()}")

        collect_phase(
            ACTIVE_TIME,
            LABELS[gesture],
            gesture.upper()
        )

        # ==================================================
        # TRANSITION TO REST
        # ==================================================
        print("\nRELAX")

        collect_phase(
            TRANSITION_TIME,
            LABELS["transition"],
            "TRANSITION"
        )

    print("\n===================================")
    print("ACQUISITION COMPLETE")
    print("===================================")

except KeyboardInterrupt:

    print("\nAcquisition interrupted by user.")

except Exception as e:

    print("\nError:", e)

finally:

    print("\nSaving data...")

    if len(samples) == 0:

        print("WARNING: No samples collected.")

    else:

        samples = np.array(samples, dtype=np.float32)
        labels = np.array(labels, dtype=np.int8)

        filename = os.path.join(
            save_dir,
            f"{gesture}_{int(time.time())}.npz"
        )

        np.savez(
            filename,
            data=samples,
            labels=labels
        )

        print(f"\nSaved {len(samples)} samples")
        print(f"Saved to:\n{filename}")