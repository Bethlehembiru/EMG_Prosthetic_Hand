# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/



# -----------------------------
# IMPORTS
# -----------------------------
from semg.acquisition.serial_reader import EMGSerialReader
from semg.buffering.buffer import EMGBuffer
from semg.preprocessing.filters import EMGFilter


# -----------------------------
# INITIALIZATION
# -----------------------------
reader = EMGSerialReader(port='COM3', baudrate=115200)

buffer = EMGBuffer(window_size=200)   # 200 samples ≈ 200 ms

filter_obj = EMGFilter(fs=1000)       # Sampling rate = 1000 Hz


# -----------------------------
# MAIN LOOP
# -----------------------------
while True:

    # Step 1: Read EMG sample
    sample = reader.read_sample()

    if sample is not None:

        # Step 2: Add to buffer
        buffer.add_sample(sample)

        # Step 3: Check if buffer is full
        if buffer.is_full():

            # Step 4: Get window
            window = buffer.get_window()

            # Step 5: Apply filtering
            filtered_signal = filter_obj.apply(window)

            # Debug output
            print("Filtered window shape:", filtered_signal.shape)
