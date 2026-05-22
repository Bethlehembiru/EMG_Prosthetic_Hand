import numpy as np

class EMGFeatureExtractor:
    def __init__(self, zc_threshold=0.01, ssc_threshold=0.01):
        # We only need thresholds for Zero Crossings and Slope Sign Changes
        self.zc_threshold = zc_threshold
        self.ssc_threshold = ssc_threshold

    def extract(self, windows):
        """
        Extracts the 4 Hudgins features (MAV, WL, ZC, SSC) from a list/array of windows.
        Only requires the filtered signal (no need to pre-rectify or calculate envelopes).
        """
        features = []

        for w in windows:
            w = np.asarray(w)
            
            # 1. Mean Absolute Value (MAV) - Amplitude estimation
            mav = np.mean(np.abs(w))
            
            # 2. Waveform Length (WL) - Signal complexity/duration
            wl = np.sum(np.abs(np.diff(w)))
            
            # 3. Zero Crossings (ZC) - Frequency estimation
            zc = np.sum(
                ((w[:-1] * w[1:]) < 0) & 
                (np.abs(w[:-1] - w[1:]) > self.zc_threshold)
            )
            
            # 4. Slope Sign Changes (SSC) - Frequency/complexity estimation
            diff1 = w[1:-1] - w[:-2]
            diff2 = w[1:-1] - w[2:]
            ssc = np.sum(
                ((diff1 * diff2) > 0) & 
                ((np.abs(diff1) > self.ssc_threshold) | (np.abs(diff2) > self.ssc_threshold))
            )
            
            features.append([mav, wl, zc, ssc])

        return np.asarray(features, dtype=np.float32)