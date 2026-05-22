import numpy as np


class EnvelopeFeatureExtractor:

    def __init__(self, fs=1000, afb_threshold=0.1):

        self.fs = fs

        self.afb_threshold = afb_threshold

    # =====================================================
    # AMPLITUDE OF FIRST BURST
    # =====================================================
    def afb(self, x):

        x = np.asarray(x)

        idx = np.where(
            x >= self.afb_threshold
        )[0]

        if len(idx) == 0:
            return [0.0]

        start = idx[0]

        end = min(
            len(x),
            start + int(0.2 * len(x))
        )

        return [np.max(x[start:end])]

    # =====================================================
    # ENVELOPE STATISTICS
    # =====================================================
    def envelope_stats(self, x):

        x = np.asarray(x)

        mean_env = np.mean(x)

        max_env = np.max(x)

        min_env = np.min(x)

        std_env = np.std(x)

        return [

            mean_env,
            max_env,
            min_env,
            std_env
        ]

    # =====================================================
    # FULL FEATURE EXTRACTION
    # =====================================================
    def extract(self, windows):

        windows = np.asarray(windows)

        features = []

        for x in windows:

            feature_vector = (

                self.afb(x) +

                self.envelope_stats(x)
            )

            features.append(feature_vector)

        return np.asarray(
            features,
            dtype=np.float32
        )