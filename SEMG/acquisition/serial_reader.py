import serial
import time


class EMGSerialReader:

    def __init__(self, port='COM3', baudrate=9600):
        self.ser = serial.Serial(port, baudrate, timeout=1)

        # Give Arduino time to reset
        time.sleep(2)
        self.ser.reset_input_buffer()

        print("Serial connection established.")

    def read_sample(self):
        try:
            line = self.ser.readline().decode(errors='ignore').strip()

            print("RAW:", line)

            if not line:
                return None

            try:
                return int(line)
            except ValueError:
                return None

        except:
            return None