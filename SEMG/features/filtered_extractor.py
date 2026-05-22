import numpy as np
from scipy.signal import welch
from statsmodels.regression.linear_model import burg


class FilteredFeatureExtractor:
    def __init__(self, fs=1000, zc_threshold=0.01,
                 wamp_threshold=0.01, ssc_threshold=0.01,
                 ar_order=4):

        self.fs = fs
        self.zc_threshold = zc_threshold
        self.wamp_threshold = wamp_threshold
        self.ssc_threshold = ssc_threshold
        self.ar_order = ar_order

    def zero_crossings(self, x):
        x = np.asarray(x)
        return [np.sum(
            ((x[:-1] * x[1:]) < 0) &
            (np.abs(x[:-1] - x[1:]) > self.zc_threshold)
        )]

    def ssc(self, x):
        x = np.asarray(x)
        diff1 = x[1:-1] - x[:-2]
        diff2 = x[1:-1] - x[2:]
        return [np.sum(
            ((diff1 * diff2) > 0) &
            ((np.abs(diff1) > self.ssc_threshold) |
             (np.abs(diff2) > self.ssc_threshold))
        )]

    def wamp(self, x):
        x = np.asarray(x)
        return [np.sum(np.abs(np.diff(x)) > self.wamp_threshold)]

    def hjorth(self, x):
        x = np.asarray(x)
        dx = np.diff(x)
        ddx = np.diff(dx)

        var_x = np.var(x)
        if var_x == 0:
            return [0.0, 0.0, 0.0]

        mobility = np.sqrt(np.var(dx) / var_x)
        complexity = (np.sqrt(np.var(ddx) / np.var(dx)) / mobility
                      if np.var(dx) > 0 else 0.0)

        return [var_x, mobility, complexity]

    def freq_features(self, x):
        x = np.asarray(x)
        f, Pxx = welch(x, fs=self.fs, nperseg=len(x))

        power = np.sum(Pxx)
        if power == 0:
            return [0, 0, 0, 0]

        mnf = np.sum(f * Pxx) / power
        cumsum = np.cumsum(Pxx)
        mdf = f[np.where(cumsum >= cumsum[-1] / 2)[0][0]]
        peak = f[np.argmax(Pxx)]

        entropy = -np.sum(
            (Pxx / power) * np.log2(Pxx / power + 1e-12)
        )

        return [mnf, mdf, peak, entropy]

    def ar_features(self, x):
        x = np.asarray(x)
        try:
            coeffs, _ = burg(x, order=self.ar_order)
            return list(coeffs)
        except:
            return [0.0] * self.ar_order

    def extract(self, windows):

        features = []

        for x in windows:
            feature_vector = (
                    self.zero_crossings(x) +
                    self.ssc(x) +
                    self.wamp(x) +
                    self.hjorth(x) +
                    self.freq_features(x) +
                    self.ar_features(x)
            )

            features.append(feature_vector)

        return np.asarray(features, dtype=np.float32)