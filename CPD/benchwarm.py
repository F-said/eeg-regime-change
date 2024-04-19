from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order

from scipy.stats import norm

from dask import delayed, compute

MAX_LAGS = 128*(1)


class ARPWarm:
    """
    Seperate logic to warmup benchmark
    """

    def __init__(self, data, n0, M, k, s=128, verbose=True):
        """
        Args:
            data    : ndarray of 12-channel eeg data
            n0      : 1s window-length during warmup
            M       : tolerated seconds of error/forecast window
            k       : halts when k channels are rejected
            s       : sampling rate of eeg (Hz)
            verbose : option to print
        """
        self.data = data
        self.n0 = n0
        self.M = M
        self.k = k
        self.sample_rate = s
        self.verbose = verbose

        self.dist_channels = {}
        self.arp_channels = {}
        self.forecast_window = M
        self.forecasts = {}
        self.trained = False

    def __warmup_chann(self, ch_data):
        """Warmup individual channel

        Args:
            ch_data :
        """
        chan_name = ch_data.name
        if self.verbose:
            print(f"Creating AR(p) for {chan_name}. Determining best order")

        # select best order via AIC
        mod = ar_select_order(ch_data, maxlag=MAX_LAGS, ic='aic')

        if self.verbose:
            print(f"Order AR({mod.ar_lags[-1]})")

        # create AR(p) model using selected order
        model = AutoReg(ch_data, lags=mod.ar_lags)
        res = model.fit()

        if self.verbose:
            print(f"Fitting complete\nAIC:{res.aic}\nLog-Likelihood:{res.llf}")

        # get gaussian mean/std of innovations
        mu, sig = norm.fit(res.resid)
        if self.verbose:
            print(f"Innovations generated\nMean:{mu}\nStd-dev:{sig}\n")

        return chan_name, res, mu, sig

    def warmup(self):
        """Step 1 of CPD baseline algorithm
        Generates best-fit AR(p) model on each channel in sample.
        Calculate and save gaussian of innovations. Operates in parallel.

        Modifies:
            dist_channels
            arp_channels
        """
        # extract sample of data
        T = self.n0 * self.sample_rate
        sample_data = self.data.iloc[:T]

        chans = sample_data.columns

        # find best lag "p" and fit AR(p) on each channel
        tasks = [delayed(self.__warmup_chann)(sample_data[ch]).values for ch in chans]
        results = compute(*tasks)

        # assign results to respective channel
        for ch, res, mu, sig in results:
            self.arp_channels[ch] = res
            self.dist_channels[ch] = (mu, sig)
            # make initial forecast to save time
            self.forecasts[ch] = res.forecast(self.forecast_window * self.sample_rate)
        self.trained = True
