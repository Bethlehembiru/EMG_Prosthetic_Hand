import serial
import time

# Change COM port (Windows: COM3, Linux/Mac: /dev/ttyUSB0 or /dev/ttyACM0)
ser = serial.Serial('COM3', 115200)
time.sleep(2)  # let Arduino reset

while True:
    cmd = input("Type 'open' or 'close': ").strip().lower()

    if cmd in ["open", "close"]:
        ser.write((cmd + "\n").encode())
    else:
        print("Invalid input. Try again.")