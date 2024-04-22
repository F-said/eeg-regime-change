from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order

from scipy.stats import norm

from dask import delayed, compute

from scipy.signal import butter, filtfilt
from sdt.changepoint import BayesOnline
import numpy as np

MAX_LAGS = 128*1
LOOKAHEAD = 128*5


class FFTWarm:
    """
    Seperate logic to warmup FFT-Bayesian Online Algorithm
    """
    def __init__(self, data, n0, chunks=4, s=128, verbose=True):
        """
        Args:
            data    : dict of 12-channel ndarray
            n0      : 1s window-length during warmup
            s       : sampling rate of eeg (Hz)
            verbose : option to print
        """
        self.data = data
        self.n0 = n0
        self.chunks = chunks
        self.sample_rate = s
        self.verbose = verbose

        self.saliency_maps = {}
        self.bayesian_cpd = {}
        self.trained = False
        self.T = self.n0 * self.sample_rate
        # init high-pass filter
        self.b, self.a = butter(
            N=2, Wn=1/(0.5*self.sample_rate),
            btype='high',
            analog=False
        )

    def __warmup_chann(self, name, data):
        """Create saliency map for individual data

        Args:
            name    : name of channel
            data    : ndarray of channel data
        """
        # compute high-pass filter
        filt_data = filtfilt(self.b, self.a, data)

        # compute FFT for channel
        magnitudes = np.fft.rfft(filt_data)

        # calculate spectral residual
        amplitudes = abs(magnitudes)
        phase = np.angle(magnitudes)

        lf = np.log(amplitudes)

        length = len(lf)
        val = (1 / length ** 2)
        q_matrix = np.full((length, length), val)
        alf = np.dot(q_matrix, lf.T)

        rf = lf - alf

        # calculate saliency map
        saliency = abs(np.fft.ifft(np.exp(rf + phase * 1j)))

        if self.verbose:
            print(f"{name}: Calculated saliency map :)")

        return name, saliency

    def __process_chunk(self, chunk):
        tasks = [delayed(self.__warmup_chann)(ch, self.data[ch][:self.T]) for ch in chunk]
        results = compute(*tasks)

        return results

    def __process_results(self, result):
        for ch, sal in result:
            self.saliency_maps[ch] = sal
            # update initial bayesian online predictor w/values (downsampling)
            bayes = BayesOnline()
            for i in range(0, len(sal), self.sample_rate // 4):
                bayes.update(sal[i])
            self.bayesian_cpd[ch] = bayes

    def warmup(self):
        """Step 1 of FFT-Bayes
        Calculate saliency maps of each channel to highlight greatest innovation.

        Modifies:
            saliency_maps
        """

        # get all channel names
        ch_names = list(self.data.keys())
        size = len(ch_names)//self.chunks

        inc = 1
        start_chunk = 0
        end_chunk = size * inc

        # process data in "chunks" chunk size
        while end_chunk <= len(ch_names):
            chunk_n = ch_names[start_chunk:end_chunk]
            print(chunk_n)
            results_n = self.__process_chunk(chunk_n)
            self.__process_results(results_n)

            start_chunk = end_chunk
            inc += 1
            end_chunk = size * inc

        self.trained = True


class ARPWarm:
    """
    Seperate logic to warmup benchmark
    """

    def __init__(self, data, n0, chunks=4, s=128, verbose=True):
        """
        Args:
            data    : dict of 12-channel ndarray
            n0      : 1s window-length during warmup
            s       : sampling rate of eeg (Hz)
            verbose : option to print
        """
        self.data = data
        self.n0 = n0
        self.chunks = chunks
        self.sample_rate = s
        self.verbose = verbose

        self.dist_channels = {}
        self.arp_channels = {}
        self.forecasts = {}
        self.trained = False
        self.T = self.n0 * self.sample_rate

    def __warmup_chann(self, name, data):
        """Warmup individual channel

        Args:
            name    : name of channel
            data    : ndarray of channel data
        """
        mod = ar_select_order(data, maxlag=MAX_LAGS, ic='aic')

        if self.verbose:
            print(f"{name}: order AR({mod.ar_lags[-1]})")

        # create AR(p) model using selected order
        model = AutoReg(data, lags=mod.ar_lags)
        res = model.fit()

        if self.verbose:
            print(f"{name}: fitting complete\nAIC:{res.aic}\nLog-Likelihood:{res.llf}")

        # get gaussian mean/std of innovations
        mu, sig = norm.fit(res.resid)
        if self.verbose:
            print(f"{name}: innovations generated\nMean:{mu}\nStd-dev:{sig}\n")

        return name, res, mu, sig

    def __process_chunk(self, chunk):
        tasks = [delayed(self.__warmup_chann)(ch, self.data[ch][:self.T]) for ch in chunk]
        results = compute(*tasks)

        return results

    def __process_results(self, result):
        for ch, res, mu, sig in result:
            self.arp_channels[ch] = res
            self.dist_channels[ch] = (mu, sig)
            # make initial forecast to save time
            self.forecasts[ch] = res.forecast(LOOKAHEAD * self.sample_rate)

    def warmup(self):
        """Step 1 of CPD baseline algorithm
        Generates best-fit AR(p) model on each channel in sample.
        Calculate and save gaussian of innovations. Operates in parallel.

        Modifies:
            dist_channels
            arp_channels
        """

        # get all channel names
        ch_names = list(self.data.keys())
        size = len(ch_names)//self.chunks

        inc = 1
        start_chunk = 0
        end_chunk = size * inc

        # process data in "chunks" chunk size
        while end_chunk <= len(ch_names):
            chunk_n = ch_names[start_chunk:end_chunk]
            print(chunk_n)
            results_n = self.__process_chunk(chunk_n)
            self.__process_results(results_n)

            start_chunk = end_chunk
            inc += 1
            end_chunk = size * inc

        self.trained = True
