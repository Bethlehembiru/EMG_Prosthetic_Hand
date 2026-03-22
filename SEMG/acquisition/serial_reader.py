import serial
import time


class EMGSerialReader:

    def __init__(self, port='COM3', baudrate=115200):

        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)

        self._initialize()

    def _initialize(self):

        print("Waiting for Arduino...")

        while True:
            line = self.ser.readline().decode().strip()

            if line == "READY":
                print("Arduino READY")
                break

        self.ser.write(b"START\n")

        while True:
            line = self.ser.readline().decode().strip()

            if line == "STREAMING":
                print("Streaming started")
                break

    def read_sample(self):

        try:
            line = self.ser.readline().decode().strip()

            if line.isdigit():
                return int(line)
            else:
                return None

        except:
            return None
