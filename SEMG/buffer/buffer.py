from collections import deque
import numpy as np


class EMGBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add_sample(self, sample):
        self.buffer.append(sample)

    def get_data(self):
        return np.array(self.buffer)

    def size(self):
        return len(self.buffer)
