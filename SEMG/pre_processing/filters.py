import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class EMGFilter:
    def __init__(self, fs=1000, envelope_cutoff=10):
        self.fs = fs
        self.envelope_cutoff = envelope_cutoff

        # Design filters once
        self._design_bandpass()
        self._design_notch()
        self._design_envelope_filter()

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
    # Envelope Low-pass Filter
    # -----------------------------
    def _design_envelope_filter(self, order=4):
        nyquist = 0.5 * self.fs
        cutoff = self.envelope_cutoff / nyquist
        self.env_b, self.env_a = butter(order, cutoff, btype='low')

    # -----------------------------
    # Apply Filtering Only
    # -----------------------------
    def apply(self, signal, return_signals=("envelope",)):
        """
        signal: 1D numpy array of raw_3_gestures EMG

        return_signals: tuple of:
            "filtered", "rectified", "envelope"

        Returns:
            tuple or single array depending on input
        """

        # Step 1: Bandpass
        filtered = filtfilt(self.bp_b, self.bp_a, signal)

        # Step 2: Notch
        filtered = filtfilt(self.notch_b, self.notch_a, filtered)

        # Step 3: Rectification
        rectified = np.abs(filtered)

        # Step 4: Envelope
        envelope = filtfilt(self.env_b, self.env_a, rectified)

        # -----------------------------
        # Return requested signals
        # -----------------------------
        output = []
        for key in return_signals:
            if key == "filtered":
                output.append(filtered)
            elif key == "rectified":
                output.append(rectified)
            elif key == "envelope":
                output.append(envelope)

        return tuple(output) if len(output) > 1 else output[0]