import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

import numpy as np
import os
from scipy.signal import butter, filtfilt, iirnotch

from SEMG.acquisition.label_and_save import labels


class EMGFilter:

    def __init__(self, fs=1000, envelope_cutoff=10, save_dir="../../data/processed"):
        self.fs = fs
        self.envelope_cutoff = envelope_cutoff
        self.save_dir = save_dir

        # Precompute filter coefficients (efficient)
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
    # Apply Full Filtering + Rectification + Envelope
    # -----------------------------
    def apply(self, signal, return_signals=("envelope",), participant=None, gesture=None):
        """
        signal: 1D numpy array of raw EMG
        return_signals: tuple, choose any of "filtered", "rectified", "envelope"
        participant: str or int, optional, used for saving
        gesture: str, optional, used for saving

        Returns:
            requested signals as tuple
        """
        # Step 1: Bandpass
        filtered = filtfilt(self.bp_b, self.bp_a, signal)

        # Step 2: Notch
        filtered = filtfilt(self.notch_b, self.notch_a, filtered)

        # Step 3: Full-wave rectification
        rectified = np.abs(filtered)

        # Step 4: Low-pass filter to get envelope
        envelope = filtfilt(self.env_b, self.env_a, rectified)

        # -----------------------------
        # Saving
        # -----------------------------
        if participant is not None and gesture is not None:
            save_subdir = os.path.join(self.save_dir, f"participant_{participant}")
            os.makedirs(save_subdir, exist_ok=True)
            filename = os.path.join(save_subdir, f"{gesture}_{int(np.floor(np.random.rand() * 1e6))}.npz")

            # Save only the envelope by default
            np.savez(filename, envelope=envelope, labels = labels)
            print(f"Processed signal saved at: {filename}")

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