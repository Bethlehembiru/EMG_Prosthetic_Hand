import numpy as np

from scipy.signal import welch
from scipy.stats import skew, kurtosis
from statsmodels.regression.linear_model import burg


class EMGFeatureExtractor:

    def __init__(
        self,
        fs=1000,
        zc_threshold=0.01,
        wamp_threshold=0.01,
        ssc_threshold=0.01,
        afb_threshold=0.1,
        ar_order=4
    ):

        self.fs = fs

        self.zc_threshold = zc_threshold
        self.wamp_threshold = wamp_threshold
        self.ssc_threshold = ssc_threshold
        self.afb_threshold = afb_threshold

        self.ar_order = ar_order

    # =====================================================
    # TIME DOMAIN FEATURES
    # BEST FOR RECTIFIED SIGNAL
    # =====================================================
    def _time_features(self, x):

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
    # ZERO CROSSINGS
    # BEST FOR FILTERED SIGNAL
    # =====================================================
    def _zero_crossings(self, x):

        x = np.asarray(x)

        zc = np.sum(

            ((x[:-1] * x[1:]) < 0) &

            (
                np.abs(x[:-1] - x[1:])
                > self.zc_threshold
            )
        )

        return [zc]

    # =====================================================
    # SLOPE SIGN CHANGES
    # =====================================================
    def _ssc(self, x):

        x = np.asarray(x)

        diff1 = x[1:-1] - x[:-2]

        diff2 = x[1:-1] - x[2:]

        ssc = np.sum(

            ((diff1 * diff2) > 0) &

            (
                (np.abs(diff1) > self.ssc_threshold) |

                (np.abs(diff2) > self.ssc_threshold)
            )
        )

        return [ssc]

    # =====================================================
    # WILLISON AMPLITUDE
    # =====================================================
    def _wamp(self, x):

        x = np.asarray(x)

        wamp = np.sum(

            np.abs(np.diff(x))
            > self.wamp_threshold
        )

        return [wamp]

    # =====================================================
    # AMPLITUDE OF FIRST BURST
    # BEST FOR ENVELOPE
    # =====================================================
    def _afb(self, x):

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
    # HJORTH PARAMETERS
    # =====================================================
    def _hjorth(self, x):

        x = np.asarray(x)

        dx = np.diff(x)

        ddx = np.diff(dx)

        var_x = np.var(x)

        if var_x == 0:
            return [0.0, 0.0, 0.0]

        activity = var_x

        mobility = np.sqrt(
            np.var(dx) / var_x
        )

        complexity = (

            np.sqrt(np.var(ddx) / np.var(dx))
            / mobility

        ) if np.var(dx) > 0 else 0.0

        return [

            activity,
            mobility,
            complexity
        ]

    # =====================================================
    # FREQUENCY DOMAIN FEATURES
    # BEST FOR FILTERED SIGNAL
    # =====================================================
    def _freq_features(self, x):

        x = np.asarray(x)

        f, Pxx = welch(

            x,

            fs=self.fs,

            nperseg=len(x)
        )

        power_sum = np.sum(Pxx)

        if power_sum == 0:

            return [
                0.0,
                0.0,
                0.0,
                0.0
            ]

        # Mean Frequency
        mnf = np.sum(f * Pxx) / power_sum

        # Median Frequency
        cumsum = np.cumsum(Pxx)

        mdf = f[
            np.where(
                cumsum >= cumsum[-1] / 2
            )[0][0]
        ]

        # Peak Frequency
        peak_freq = f[np.argmax(Pxx)]

        # Spectral Entropy
        p_norm = Pxx / power_sum

        spectral_entropy = -np.sum(

            p_norm *
            np.log2(p_norm + 1e-12)
        )

        return [

            mnf,
            mdf,
            peak_freq,
            spectral_entropy
        ]

    # =====================================================
    # AUTO REGRESSIVE FEATURES
    # =====================================================
    def _ar_features(self, x):

        x = np.asarray(x)

        try:

            ar_coeffs, _ = burg(

                x,

                order=self.ar_order
            )

            return list(ar_coeffs)

        except:

            return [0.0] * self.ar_order

    # =====================================================
    # ENVELOPE FEATURES
    # =====================================================
    def _envelope_features(self, x):

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
    def extract(

        self,

        filtered_windows,

        rectified_windows,

        envelope_windows
    ):

        features = []

        for fw, rw, ew in zip(

            filtered_windows,

            rectified_windows,

            envelope_windows
        ):

            fw = np.asarray(fw)

            rw = np.asarray(rw)

            ew = np.asarray(ew)

            # =================================================
            # FILTERED FEATURES
            # =================================================
            filtered_features = (

                self._zero_crossings(fw) +

                self._ssc(fw) +

                self._wamp(fw) +

                self._hjorth(fw) +

                self._freq_features(fw) +

                self._ar_features(fw)
            )

            # =================================================
            # RECTIFIED FEATURES
            # =================================================
            rectified_features = (

                self._time_features(rw)
            )

            # =================================================
            # ENVELOPE FEATURES
            # =================================================
            envelope_features = (

                self._afb(ew) +

                self._envelope_features(ew)
            )

            # =================================================
            # FINAL FEATURE VECTOR
            # =================================================
            feature_vector = (

                filtered_features +

                rectified_features +

                envelope_features
            )

            features.append(feature_vector)

        return np.asarray(

            features,

            dtype=np.float32
        )