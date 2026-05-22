import numpy as np
from scipy.stats import skew, kurtosis


class RectifiedFeatureExtractor:

    def __init__(self, fs=1000):
        self.fs = fs

    # =====================================================
    # TIME DOMAIN FEATURES
    # =====================================================
    def time_features(self, x):

        x = np.asarray(x)

        min_v = np.min(x)

        max_v = np.max(x)

        mean_v = np.mean(x)

        std = (
            np.std(x, ddof=1)
            if len(x) > 1 else 0.0
        )

        var = np.var(x)

        rms = np.sqrt(np.mean(x ** 2))

        mav = np.mean(np.abs(x))

        iemg = np.sum(np.abs(x))

        wl = np.sum(np.abs(np.diff(x)))

        aac = (
            np.mean(np.abs(np.diff(x)))
            if len(x) > 1 else 0.0
        )

        skewness = skew(x)

        kurt = kurtosis(x)

        return [

            min_v,
            max_v,
            mean_v,

            std,
            var,

            rms,
            mav,
            iemg,

            wl,
            aac,

            skewness,
            kurt
        ]

    # =====================================================
    # FULL FEATURE EXTRACTION
    # =====================================================
    def extract(self, windows):

        windows = np.asarray(windows)

        features = []

        for x in windows:

            feature_vector = self.time_features(x)

            features.append(feature_vector)

        return np.asarray(
            features,
            dtype=np.float32
        )