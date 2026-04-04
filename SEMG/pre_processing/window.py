import numpy as np


class WindowSegmenter:
    def __init__(self, window_size=250, overlap=0.5, reject_label=-1):
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.reject_label = reject_label

    def segment(self, signal, labels=None):
        """
        Works for:
        - Training (labels provided)
        - Inference (labels=None)
        """

        windows = []
        window_labels = []

        start = 0

        while start + self.window_size <= len(signal):
            end = start + self.window_size
            window = signal[start:end]

            # -------------------------
            # TRAINING MODE (with labels)
            # -------------------------
            if labels is not None:
                window_lbl = labels[start:end]

                # Reject transition
                if self.reject_label in window_lbl:
                    start += self.step_size
                    continue

                # Reject mixed labels
                if len(np.unique(window_lbl)) > 1:
                    start += self.step_size
                    continue

                final_label = window_lbl[0]
                windows.append(window)
                window_labels.append(final_label)

            # -------------------------
            # INFERENCE MODE (no labels)
            # -------------------------
            else:
                windows.append(window)

            start += self.step_size

        # Return format depends on mode
        if labels is not None:
            return np.array(windows), np.array(window_labels)
        else:
            return np.array(windows), None