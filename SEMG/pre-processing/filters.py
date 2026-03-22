import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class EMGFilter:

    def __init__(self, fs=1000):
        self.fs = fs

        # Precompute filter coefficients (efficient)
        self._design_bandpass()
        self._design_notch()

    # -----------------------------
    # Bandpass Filter (20–450 Hz)
    # -----------------------------
    def _design_bandpass(self, lowcut=20, highcut=450, order=4):
        
        nyquist = 0.5 * self.fs
        
        low = lowcut / nyquist
        high = highcut / nyquist
        
        self.bp_b, self.bp_a = butter(order, [low, high], btype='band')

    # -----------------------------
    # Notch Filter (50 Hz)
    # -----------------------------
    def _design_notch(self, freq=50, Q=30):
        
        self.notch_b, self.notch_a = iirnotch(freq, Q, self.fs)

    # -----------------------------
    # Apply Full Filtering Pipeline
    # -----------------------------
    def apply(self, signal):

        # Step 1: Bandpass
        filtered = filtfilt(self.bp_b, self.bp_a, signal)

        # Step 2: Notch
        filtered = filtfilt(self.notch_b, self.notch_a, filtered)

        return filtered
