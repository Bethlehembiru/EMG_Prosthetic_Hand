import time
import serial
import joblib
import numpy as np

from collections import deque

from SEMG.features.extractor import EMGFeatureExtractor
from SEMG.pre_processing.filters import EMGFilter
from SEMG.control.hand_control import HandController


# =========================================================
# SETTINGS
# =========================================================
MODEL_PATH = r"C:\Users\lukma\Desktop\hand\EMG_Prosthetic_Hand\model_train_test2\best_model.joblib"

SERIAL_PORT = "COM4"
BAUDRATE = 9600

FS = 1000
WINDOW_SIZE = 250
STEP_SIZE = 125   # 50% overlap

SMOOTHING = 5


# =========================================================
# LOAD MODEL
# =========================================================
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")


# =========================================================
# PIPELINE OBJECTS
# =========================================================
filter_bank = EMGFilter(fs=FS)
extractor = EMGFeatureExtractor(fs=FS)

controller = HandController(
    port=SERIAL_PORT,
    baudrate=BAUDRATE
)


# =========================================================
# SERIAL CONNECTION
# =========================================================
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
time.sleep(2)

ser.reset_input_buffer()

print("Connected to EMG stream.")


# =========================================================
# BUFFERS
# =========================================================
buffer = deque(maxlen=WINDOW_SIZE * 5)
pred_buffer = deque(maxlen=SMOOTHING)

sample_counter = 0


# =========================================================
# STREAM LOOP
# =========================================================
print("\nRunning online inference...\n")

try:
    while True:

        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        # =====================================================
        # ONLY ACCEPT EMG LINES (IMPORTANT)
        # =====================================================
        if not line.startswith("E:"):
            continue

        try:
            sample = float(line.replace("E:", ""))
        except:
            continue

        buffer.append(sample)
        sample_counter += 1

        # wait until enough data
        if len(buffer) < WINDOW_SIZE:
            continue

        # step control
        if sample_counter < STEP_SIZE:
            continue

        sample_counter = 0

        # =====================================================
        # GET WINDOW (latest segment only)
        # =====================================================
        window = np.array(list(buffer)[-WINDOW_SIZE:])

        # =====================================================
        # FILTERING (same as training pipeline)
        # =====================================================
        filtered, rectified, envelope = filter_bank.apply(
            window,
            return_signals=("filtered", "rectified", "envelope")
        )

        # =====================================================
        # FEATURE EXTRACTION (31 features_close_normalized)
        # =====================================================
        features = extractor.extract(
            filtered.reshape(1, -1),
            rectified.reshape(1, -1),
            envelope.reshape(1, -1)
        )

        # =====================================================
        # PREDICTION
        # =====================================================
        pred = model.predict(features)[0]

        pred_buffer.append(pred)

        final_pred = max(set(pred_buffer), key=pred_buffer.count)

        print(f"Raw: {pred} | Smoothed: {final_pred}")

        # =====================================================
        # SEND TO ARDUINO (C: prefix handled in controller)
        # =====================================================
        controller.send_gesture(final_pred)


except KeyboardInterrupt:
    print("\nStopping...")

finally:
    ser.close()
    controller.close()
    print("Closed.")