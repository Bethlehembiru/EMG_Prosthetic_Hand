import time


class HandController:

    def __init__(
        self,
        serial_connection,
        min_command_interval=0.05
    ):

        # =====================================================
        # SHARED SERIAL CONNECTIO
        # =====================================================
        self.ser = serial_connection

        # =====================================================
        # TIMING CONTROL
        # =====================================================
        self.last_send_time = 0
        self.min_command_interval = min_command_interval

        self.last_prediction = None

        # =====================================================
        # GESTURE MAP
        # =====================================================
        self.gesture_map = {

            1: {  # CLOSE
                "thumb": 180,
                "index": 180,
                "middle": 0,
                "ring": 180,
                "pinky": 0
            },

            0:  {  # rest
                "thumb": 0,
                "index": 0,
                "middle": 180,
                "ring": 0,
                "pinky": 180
            },
 
        }

        print("\nHand Controller Ready")

    # =====================================================
    # SEND GESTURE
    # =====================================================
    def send_gesture(self, prediction):

        if prediction not in self.gesture_map:
            return

        # avoid duplicate commands
        if prediction == self.last_prediction:
            return

        # rate limiting
        current_time = time.time()

        if current_time - self.last_send_time < self.min_command_interval:
            return

        g = self.gesture_map[prediction]

        command = (
            f"C:{g['thumb']},"
            f"{g['index']},"
            f"{g['middle']},"
            f"{g['ring']},"
            f"{g['pinky']}\n"
        )

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