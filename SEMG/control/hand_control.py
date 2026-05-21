import serial
import time


class HandController:

    def __init__(
        self,
        port="COM4",
        baudrate=9600,
        timeout=1,
        min_command_interval=0.05
    ):

        # =====================================================
        # SERIAL CONNECTION
        # =====================================================
        self.ser = serial.Serial(
            port,
            baudrate,
            timeout=timeout
        )

        time.sleep(2)

        # flush startup garbage
        self.ser.reset_input_buffer()

        # =====================================================
        # TIMING CONTROL
        # =====================================================
        self.last_send_time = 0
        self.min_command_interval = min_command_interval

        self.last_prediction = None

        # =====================================================
        # GESTURE MAP (5 motors)
        # =====================================================
        self.gesture_map = {

            0: {  # REST
                "thumb": 90,
                "index": 90,
                "middle": 90,
                "ring": 90,
                "pinky": 90
            },

            1: {  # CLOSE
                "thumb": 180,
                "index": 180,
                "middle": 0,
                "ring": 180,
                "pinky": 0
            },

            2: {  # OPEN
                "thumb": 0,
                "index": 0,
                "middle": 180,
                "ring": 0,
                "pinky": 180
            },

            3: {  # POINT
                "thumb": 180,
                "index": 0,
                "middle": 0,
                "ring": 180,
                "pinky": 0
            }
        }

        print("\nHand Controller Connected")
        print(f"Port: {port}")
        print(f"Baudrate: {baudrate}")

    # =====================================================
    # SEND GESTURE
    # =====================================================
    def send_gesture(self, prediction):

        if prediction not in self.gesture_map:
            return

        # avoid spam repetition
        if prediction == self.last_prediction:
            return

        # rate limit
        current_time = time.time()
        if current_time - self.last_send_time < self.min_command_interval:
            return

        g = self.gesture_map[prediction]

        thumb = g["thumb"]
        index = g["index"]
        middle = g["middle"]
        ring = g["ring"]
        pinky = g["pinky"]

        # =====================================================
        # IMPORTANT: PREFIX "C:"
        # =====================================================
        command = f"C:{thumb},{index},{middle},{ring},{pinky}\n"

        try:
            self.ser.write(command.encode())

            self.last_prediction = prediction
            self.last_send_time = current_time

            print(f"[CMD] {prediction} -> {command.strip()}")

        except Exception as e:
            print("Serial write error:", e)

    # =====================================================
    # OPTIONAL MANUAL CONTROL
    # =====================================================
    def send_angles(self, thumb, index, middle, ring, pinky):

        command = f"C:{thumb},{index},{middle},{ring},{pinky}\n"

        try:
            self.ser.write(command.encode())
            print("[MANUAL]", command.strip())

        except Exception as e:
            print("Serial write error:", e)

    # =====================================================
    # CLOSE
    # =====================================================
    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")