from collections import deque
import numpy as np


class EMGBuffer:

    def __init__(self, window_size=200):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add_sample(self, sample):
        self.buffer.append(sample)

    def is_full(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        return np.array(self.buffer)
