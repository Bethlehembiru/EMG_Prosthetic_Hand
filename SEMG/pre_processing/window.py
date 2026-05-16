import numpy as np


class WindowSegmenter:
    def __init__(self, window_size=250, overlap=0.5, reject_label=-1):
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.reject_label = reject_label

    def segment(self, signal, labels=None):
        windows = []
        window_labels = []

        rejected_stats = {
            "transition_rejected": 0,
            "mixed_label_rejected": 0,
            "total_windows": 0
        }

        start = 0

        while start + self.window_size <= len(signal):
            end = start + self.window_size

            window = signal[start:end]
            rejected_stats["total_windows"] += 1

            if labels is not None:
                window_lbl = labels[start:end]

                # reject transition regions
                if self.reject_label in window_lbl:
                    rejected_stats["transition_rejected"] += 1
                    start += self.step_size
                    continue

                # reject mixed-label windows
                if len(np.unique(window_lbl)) > 1:
                    rejected_stats["mixed_label_rejected"] += 1
                    start += self.step_size
                    continue

                final_label = window_lbl[0]
                windows.append(window)
                window_labels.append(final_label)

            else:
                windows.append(window)

            start += self.step_size

        if labels is not None:
            return (
                np.asarray(windows),
                np.asarray(window_labels),
                rejected_stats
            )

        return np.asarray(windows), None, rejected_stats