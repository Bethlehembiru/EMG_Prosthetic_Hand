import time

# Core modules
from SEMG.acquisition.serial_reader import EMGSerialReader
from SEMG.buffer.buffer import EMGBuffer
from SEMG.pre_processing.filters import EMGFilter
from SEMG.pre_processing.window import WindowSegmenter
from SEMG.features.extractor import EMGFeatureExtractor

# Model
from joblib import load


from SEMG.control.hand_control import HandController


# ----------------------------
# Load trained model
# ----------------------------
model = load("models/svm.joblib")  # or lda / knn


# ----------------------------
# Initialize components
# ----------------------------
reader = EMGSerialReader()
buffer = EMGBuffer(max_size=10000)
filter_ = EMGFilter()
windower = WindowSegmenter(window_size=250, overlap=0.5)
feature_extractor = EMGFeatureExtractor()

# Hand controller (Arduino bridge)
controller = HandController(port="COM3")


# ----------------------------
# Real-time loop
# ----------------------------
print("Starting real-time EMG control...\n")

try:
    while True:
        # 1. Read signal
        sample = reader.read_sample()

        # 2. Buffer it
        buffer.add_sample(sample)

        # 3. Wait for enough data
        if buffer.size() < windower.window_size:
            continue

        data = buffer.get_data()

        # 4. Filtering
        filtered = filter_.apply(data)

        # 5. Windowing
        windows, _ = windower.segment(filtered)

        if len(windows) == 0:
            continue

        # 6. Feature extraction
        features_batch = feature_extractor.extract(windows)

        # 7. Prediction + control
        # Use only the latest window to avoid spamming
        features = features_batch[-1]

        pred = model.predict(features.reshape(1, -1))[0]

        print("Final Prediction:", pred)

        # Ignore transition class if used
        if pred != -1:
            controller.send_gesture(pred)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopped.")