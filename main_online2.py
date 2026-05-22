import time
import serial
import joblib
import numpy as np

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
STEP_SIZE = 125

SMOOTHING = 3   # reduced latency


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
# SERIAL (non-blocking style)
# =========================================================
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0)
time.sleep(2)
ser.reset_input_buffer()

print("Connected to EMG stream.")


# =========================================================
# FAST RING BUFFER (NO deque slicing)
# =========================================================
buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
idx = 0
filled = False

sample_counter = 0
pred_buffer = []


# =========================================================
# STREAM LOOP
# =========================================================
print("\nRunning FAST online inference...\n")

try:
    while True:

        line = ser.readline().decode(errors="ignore").strip()

        if not line or not line.startswith("E:"):
            continue

        try:
            sample = float(line[2:])  # faster than replace()
        except:
            continue

        # =====================================================
        # RING BUFFER UPDATE (FAST)
        # =====================================================
        buffer[idx] = sample
        idx = (idx + 1) % WINDOW_SIZE

        if idx == 0:
            filled = True

        sample_counter += 1

        if not filled:
            continue

        # STEP CONTROL
        if sample_counter < STEP_SIZE:
            continue

        sample_counter = 0


        # =====================================================
        # FAST WINDOW RECONSTRUCTION (NO list(), NO slicing)
        # =====================================================
        window = np.roll(buffer, -idx).copy()


        # =====================================================
        # FILTERING (still expensive, but optimized call)
        # =====================================================
        filtered, rectified, envelope = filter_bank.apply(
            window,
            return_signals=("filtered", "rectified", "envelope")
        )


        # =====================================================
        # FEATURE EXTRACTION (MAIN BOTTLENECK)
        # =====================================================
        features = extractor.extract(
            filtered,
            rectified,
            envelope
        )

        # ensure correct shape for sklearn
        features = np.asarray(features).reshape(1, -1)


        # =====================================================
        # PREDICTION
        # =====================================================
        pred = model.predict(features)[0]

        pred_buffer.append(pred)

        if len(pred_buffer) > SMOOTHING:
            pred_buffer.pop(0)

        final_pred = max(set(pred_buffer), key=pred_buffer.count)


        print(f"Raw: {pred} | Smoothed: {final_pred}")


        # =====================================================
        # SEND TO ARDUINO
        # =====================================================
        controller.send_gesture(final_pred)


except KeyboardInterrupt:
    print("\nStopping...")

finally:
    ser.close()
    controller.close()
    print("Closed.")