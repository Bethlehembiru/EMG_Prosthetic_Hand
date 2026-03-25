import numpy as np
import os
from scipy.signal import welch

class EMGFeatureExtractor:
    """
    Feature extraction for EMG signals with windowing.
    Saves feature vectors and corresponding labels in data/features/.
    """

    def __init__(self, window_size=250, overlap=0.5, fs=1000):
        """
        window_size : int
            Number of samples per window
        overlap : float
            Fraction of window overlap (0-1)
        fs : int
            Sampling frequency (Hz)
        """
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap))
        self.fs = fs
        self.save_dir = "data/features"
        os.makedirs(self.save_dir, exist_ok=True)

    # -----------------------------
    # Time-domain features
    # -----------------------------
    def _time_domain_features(self, window):
        mav = np.mean(np.abs(window))  # Mean Absolute Value
        rms = np.sqrt(np.mean(window ** 2))  # Root Mean Square
        wl = np.sum(np.abs(np.diff(window)))  # Waveform Length
        zc = np.sum(((window[:-1] * window[1:]) < 0) & (np.abs(np.diff(window)) > 1e-6))  # Zero crossings
        ssc = np.sum(((np.diff(window[:-1]) * np.diff(window[1:])) < 0))  # Slope sign changes
        iemg = np.sum(np.abs(window))  # Integrated EMG
        return [mav, rms, wl, zc, ssc, iemg]

    # -----------------------------
    # Frequency-domain features
    # -----------------------------
    def _frequency_domain_features(self, window):
        f, Pxx = welch(window, fs=self.fs, nperseg=len(window))
        mnf = np.sum(f * Pxx) / np.sum(Pxx)  # Mean frequency
        cumsum = np.cumsum(Pxx)
        total = cumsum[-1]
        mdf_idx = np.where(cumsum >= total / 2)[0][0]
        mdf = f[mdf_idx]  # Median frequency
        ttp = f[np.argmax(Pxx)]  # Peak frequency
        return [mnf, mdf, ttp]

    # -----------------------------
    # Main feature extraction
    # -----------------------------
    def extract_and_save(self, signal, labels, participant=None, gesture=None,
                         use_freq=True):
        """
        signal : 1D np.array
            Processed EMG envelope
        labels : 1D np.array
            Corresponding labels for each sample
        participant : str
            Participant ID
        gesture : str
            Gesture name (for saving)
        use_freq : bool
            Whether to include frequency-domain features
        """
        features = []
        feature_labels = []

        start = 0
        while start + self.window_size <= len(signal):
            window = signal[start:start + self.window_size]
            window_labels = labels[start:start + self.window_size]

            # Reject window if it contains multiple labels
            if len(np.unique(window_labels)) > 1:
                start += self.step_size
                continue

            # Extract features
            feat_td = self._time_domain_features(window)
            feat_fd = self._frequency_domain_features(window) if use_freq else []
            feat_vector = feat_td + feat_fd

            features.append(feat_vector)
            feature_labels.append(window_labels[0])  # All labels same, pick first

            start += self.step_size

        features = np.array(features)
        feature_labels = np.array(feature_labels)

        # -----------------------------
        # Save feature vectors
        # -----------------------------
        if participant is not None and gesture is not None:
            save_subdir = os.path.join(self.save_dir, f"participant_{participant}")
            os.makedirs(save_subdir, exist_ok=True)
            filename = os.path.join(save_subdir, f"{gesture}_{int(np.floor(np.random.rand()*1e6))}.npz")
            np.savez(filename, features=features, labels=feature_labels)
            print(f"Feature vectors saved at: {filename}")

        return features, feature_labels
