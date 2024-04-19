from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order

from scipy.stats import norm

from dask import delayed, compute

MAX_LAGS = 128*3
LOOKAHEAD = 10

class ARPWarm:
    """
    Seperate logic to warmup benchmark
    """

    def __init__(self, data, n0, M, s=128, verbose=True):
        """
        Args:
            data    : dict of 12-channel ndarray
            n0      : 1s window-length during warmup
            M       : tolerated seconds of error/forecast window
            s       : sampling rate of eeg (Hz)
            verbose : option to print
        """
        self.data = data
        self.n0 = n0
        self.M = M
        self.sample_rate = s
        self.verbose = verbose

        self.dist_channels = {}
        self.arp_channels = {}
        self.forecast_window = M * LOOKAHEAD
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
            self.forecasts[ch] = res.forecast(self.forecast_window * self.sample_rate)

    def warmup(self):
        """Step 1 of CPD baseline algorithm
        Generates best-fit AR(p) model on each channel in sample.
        Calculate and save gaussian of innovations. Operates in parallel.

        Modifies:
            dist_channels
            arp_channels
        """
        # TODO: a lot of repeating and perhaps improper usage, refactor
        # calc sample of data
        T = self.n0 * self.sample_rate

        # get all channel names
        ch_names = list(self.data.keys())
        # split into 4 partitions for processing
        chan1 = ch_names[:3]
        chan2 = ch_names[3:6]
        chan3 = ch_names[6:9]
        chan4 = ch_names[9:]

        # find best lag "p" and fit AR(p) on each channel (in parallel chunks)
        results1 = self.__process_chunk(chan1)
        results2 = self.__process_chunk(chan2)
        results3 = self.__process_chunk(chan3)
        results4 = self.__process_chunk(chan4)

        # assign results to respective channel
        self.__process_results(results1)
        self.__process_results(results2)
        self.__process_results(results3)
        self.__process_results(results4)

        self.trained = True
