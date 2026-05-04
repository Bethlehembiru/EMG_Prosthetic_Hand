import serial
import time

port = 'COM3'   # change if needed
baud_rate = 115200

ser = serial.Serial(port, baud_rate)
time.sleep(2)  # Arduino resets when connected

while True:
    line = ser.readline().decode('utf-8').strip()
    if line:
        try:
            value = int(line)
            print(value)
        except:
            pass