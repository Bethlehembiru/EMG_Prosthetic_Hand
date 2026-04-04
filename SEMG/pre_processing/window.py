import numpy as np


class WindowSegmenter:
    def __init__(self, window_size=250, overlap=0.5, reject_label=-1):
        """
        window_size : int
            Number of samples per window
        overlap : float
            Fraction of overlap (0–1)
        reject_label : int
            Label to reject (e.g., transition = -1)
        """
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.reject_label = reject_label

    def segment(self, signal, labels = None):
        """
        Segment signal into overlapping windows and assign labels.

        Parameters:
            signal : 1D numpy array
            labels : 1D numpy array

        Returns:
            windows : (N, window_size)
            window_labels : (N,)
        """

        windows = []
        window_labels = []

        start = 0

        while start + self.window_size <= len(signal):
            end = start + self.window_size

            window = signal[start:end]
            window_lbl = labels[start:end]

            # Reject transition windows
            if self.reject_label in window_lbl:
                start += self.step_size
                continue

            # Reject mixed labels (clean training data)
            if len(np.unique(window_lbl)) > 1:
                start += self.step_size
                continue

            # All labels same → pick first
            final_label = window_lbl[0]

            windows.append(window)
            window_labels.append(final_label)

            start += self.step_size

        return np.array(windows), np.array(window_labels)