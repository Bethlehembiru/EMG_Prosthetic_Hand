from semg.acquisition.serial_reader import SerialReader
from semg.buffer.buffer import Buffer
from semg.preprocessing.filters import apply_filters
from semg.features.extractor import extract_features
from semg.classification.classifier import Classifier
from semg.control.controller import Controller
from semg.utils.config import *


def main():
    print("Starting sEMG pipeline...")

    # Initialize components
    reader = SerialReader(PORT, BAUD_RATE)
    buffer = Buffer(BUFFER_SIZE)
    classifier = Classifier(MODEL_PATH)
    controller = Controller()

    try:
        while True:
            # Step 1: Acquire data
            data = reader.read()

            # Step 2: Buffer data
            buffer.add(data)

            if buffer.is_full():
                window = buffer.get_window()

                # Step 3: Preprocess
                processed = apply_filters(window)

                # Step 4: Feature extraction
                features = extract_features(processed)

                # Step 5: Classification
                prediction = classifier.predict(features)

                print(f"Prediction: {prediction}")

                # Step 6: Control
                controller.execute(prediction)

    except KeyboardInterrupt:
        print("Stopping pipeline...")

    finally:
        reader.close()


if __name__ == "__main__":
    main()
