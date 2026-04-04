import serial
import time


class HandController:
    def __init__(self, port="COM3", baudrate=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)

        # Full gesture → multi-servo mapping
        self.gesture_map = {
            0: {"thumb": 90, "index": 90, "multi": 90},   # Rest
            1: {"thumb": 0, "index": 0, "multi": 0},       # Close
            2: {"thumb": 180, "index": 180, "multi": 180}, # Open
            3: {"thumb": 45, "index": 0, "multi": 90}      # Point
        }

    def send_gesture(self, prediction):
        if prediction not in self.gesture_map:
            return

        gesture = self.gesture_map[prediction]

        # Send as a structured string
        command = f"{gesture['thumb']},{gesture['index']},{gesture['multi']}\n"
        self.ser.write(command.encode())

        print(f"Prediction: {prediction} → {gesture}")

    def close(self):
        self.ser.close()