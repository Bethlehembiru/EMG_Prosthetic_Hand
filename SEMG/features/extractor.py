import numpy as np
from scipy.signal import welch


class EMGFeatureExtractor:
    def __init__(self, fs=1000):
        self.fs = fs

    # ---- Time features (3) ----
    def _time_features(self, window):
        mav = np.mean(np.abs(window))
        rms = np.sqrt(np.mean(window ** 2))
        wl = np.sum(np.abs(np.diff(window)))
        return [mav, rms, wl]

    # ---- Frequency features (3) ----
    def _freq_features(self, window):
        f, Pxx = welch(window, fs=self.fs, nperseg=len(window))

        mnf = np.sum(f * Pxx) / np.sum(Pxx)

        cumsum = np.cumsum(Pxx)
        total = cumsum[-1]
        mdf = f[np.where(cumsum >= total / 2)[0][0]]

        ttp = f[np.argmax(Pxx)]

        return [mnf, mdf, ttp]

    # ---- Main extraction ----
    def extract(self, windows):
        features = []

        for window in windows:
            td = self._time_features(window)
            fd = self._freq_features(window)
            features.append(td + fd)

        return np.array(features)