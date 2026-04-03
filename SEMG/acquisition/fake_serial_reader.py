import numpy as np
import time

class EMGSerialReader:
    def __init__(self, port=None, baudrate=115200):
        self.fs = 1000
        self.start_time = time.time()

    def read_sample(self):
        t = time.time() - self.start_time

        # simulate EMG-like signal (noise + bursts)
        base = np.random.randn() * 0.05

        # simulate "muscle activity" cycles
        cycle_time = 3 + 0.5 + 3 + 0.5
        phase = t % cycle_time

        if 3.5 < phase < 6.5:
            signal = base + np.random.randn() * 0.5  # active
        else:
            signal = base  # rest / transition

        return signal