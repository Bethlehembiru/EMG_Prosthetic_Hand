import time
import serial
import joblib
import numpy as np

from collections import deque, Counter

from SEMG.features.extractor import EMGFeatureExtractor
from SEMG.pre_processing.filters import EMGFilter
from SEMG.control.hand_controller_grip import HandController


# =========================================================
# SETTINGS
# =========================================================
MODEL_PATH = r"C:\Users\lukma\Desktop\hand\EMG_Prosthetic_Hand\models\best_qda_model.joblib"

SERIAL_PORT = "COM4"
BAUDRATE = 9600

FS = 1000

# lower latency
WINDOW_SIZE = 150
STEP_SIZE = 75

# smoothing
SMOOTHING = 1


# =========================================================
# LOAD MODEL
# =========================================================
print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")


# =========================================================
# SERIAL CONNECTION (ONLY ONCE)
# =========================================================
print("Opening serial connection...")

ser = serial.Serial(
    SERIAL_PORT,
    BAUDRATE,
    timeout=1
)

time.sleep(2)

ser.reset_input_buffer()

print("Serial connected.")


# =========================================================
# PIPELINE OBJECTS
# =========================================================
filter_bank = EMGFilter(fs=FS)

extractor = EMGFeatureExtractor(fs=FS)

# shared serial object
controller = HandController(
    serial_connection=ser
)


# =========================================================
# BUFFERS
# =========================================================
buffer = deque(maxlen=WINDOW_SIZE)

pred_buffer = deque(maxlen=SMOOTHING)

sample_counter = 0


# =========================================================
# STREAM LOOP
# =========================================================
print("\nRunning online inference...\n")

try:

    while True:

        # =====================================================
        # READ SERIAL
        # =====================================================
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        # =====================================================
        # ACCEPT ONLY EMG DATA
        # =====================================================
        if not line.startswith("E:"):
            continue

        try:

            sample = float(line.replace("E:", ""))

        except:

            continue

        # =====================================================
        # BUFFERING
        # =====================================================
        buffer.append(sample)

        sample_counter += 1

        # wait for full window
        if len(buffer) < WINDOW_SIZE:
            continue

        # overlap control
        if sample_counter < STEP_SIZE:
            continue

        sample_counter = 0

        # =====================================================
        # GET WINDOW
        # =====================================================
        window = np.array(buffer)

        # =====================================================
        # FILTERING
        # =====================================================
        filtered, rectified, envelope = filter_bank.apply(
            window,
            return_signals=(
                "filtered",
                "rectified",
                "envelope"
            )
        )

        # =====================================================
        # FEATURE EXTRACTION (MATCH TRAINING)
        # FEATURES:
        # MAV, WL, ZC, SSC
        # =====================================================

        # ---------- MAV ----------
        mav = np.mean(np.abs(rectified))

        # ---------- WL ----------
        wl = np.sum(np.abs(np.diff(filtered)))

        # ---------- ZC ----------
        zc_threshold = 0.01

        zc = np.sum(
            (
                (filtered[:-1] * filtered[1:]) < 0
            ) &
            (
                np.abs(filtered[:-1] - filtered[1:]) >= zc_threshold
            )
        )

        # ---------- SSC ----------
        ssc_threshold = 0.01

        diff1 = np.diff(filtered)

        ssc = np.sum(
            (
                (diff1[:-1] * diff1[1:]) < 0
            ) &
            (
                np.abs(diff1[:-1] - diff1[1:]) >= ssc_threshold
            )
        )

        # =====================================================
        # FINAL FEATURE VECTOR
        # IMPORTANT:
        # MUST MATCH TRAINING ORDER EXACTLY
        # =====================================================

        features = np.array([
            mav,
            wl,
            zc,
            ssc
        ], dtype=np.float32).reshape(1, -1)

        # =====================================================
        # PREDICTION
        # =====================================================

        pred = model.predict(features)[0]

        # =====================================================
        # PREDICTION
        # =====================================================
        pred = model.predict(features)[0]

        # =====================================================
        # MAJORITY VOTE SMOOTHING
        # =====================================================
        pred_buffer.append(pred)

        counts = Counter(pred_buffer)

        final_pred = counts.most_common(1)[0][0]

        print(f"Raw: {pred} | Smoothed: {final_pred}")

        # =====================================================
        # SEND TO HAND
        # =====================================================
        controller.send_gesture(final_pred)

except KeyboardInterrupt:

    print("\nStopping...")

finally:

    ser.close()

    print("Serial connection closed.")