from scipy.stats import norm, kstest
import numpy as np

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import ar_select_order

from dask import delayed, compute
from dask.distributed import Client, Variable

MAX_LAGS = 128*2
SIGMA = 5


class Online:
    """
    Benchmark change-point-detection algorithm
    """

    def __init__(self, data, n0, M, k, s=128, verbose=True):
        """
        Args:
            data    : ndarray of 12-channel eeg data
            n0      : 1s window-length during warmup
            M       : M...
            k       : k channel heuristic
            s       : sampling rate of eeg
            verbose : option to print
        """
        self.data = data
        self.n0 = n0
        self.M = M
        self.k = k
        self._sample_rate = s
        self.verbose = verbose

        self._dist_channels = {}
        self._arp_channels = {}

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
            _dist_channels
            _arp_channels
        """
        # extract sample of data
        T = self.n0 * self._sample_rate
        sample_data = self.data.iloc[:T]

        chans = sample_data.columns

        # find best lag "p" and fit AR(p) on each channel
        tasks = [delayed(self.__warmup_chann)(sample_data[ch]) for ch in chans]
        results = compute(*tasks)

        # assign results to respective channel
        for ch, res, mu, sig in results:
            self._arp_channels[ch] = res
            self._dist_channels[ch] = (mu, sig)

    def __predict(self, ch, step):
        """Step 2 of CPD baseline algorithm
        Predicts data in next window of size s

        Returns:
            innovations : prediction error on next window
        """
        # calculate next window of data (+1 second)
        window = step * self._sample_rate
        # exit if beyond samples
        if window > self.data.shape[0]:
            return None

        # predict data in next sample rate window using AR(p) results
        next_forecast = (step - self.n0) + 1
        pred = self._arp_channels[ch].forecast(next_forecast * self._sample_rate)
        pred = pred.iloc[-self._sample_rate:]

        # calculate innovations in this window by comparing actual data
        next_data = self.data.loc[window:window + (self._sample_rate - 1), ch]
        innovations = next_data - pred

        return innovations.values

    def __detect_chann(self, ch, step):
        """Detect change-point on individual channel
        """
        rejection_count = 0
        next_innov = 0

        while next_innov is not None:

            if self.halted.get() >= self.k:
                if self.verbose:
                    print(f"{self.k} channels halted. Halting detection.")
                break

            if self.verbose:
                print(f"{ch}: checking for change-point at step {step}")

            next_innov = self.__predict(ch, step)

            if next_innov is None:
                if self.verbose:
                    print(f"No more data! Stopped at step {step}")
                    step = 0
                break

            # calc 5-sig
            mu, sig = self._dist_channels[ch]
            x = np.mean(next_innov)
            z_score = abs((x - mu)/sig)

            # check for deviation greater than N-sigma
            if z_score >= SIGMA:
                if self.verbose:
                    print(f"{ch} Rejection incremented")
                rejection_count += 1
                # halt if number of rejects exceeds M
                if self.__decide(rejection_count):
                    if self.verbose:
                        print(f"{ch} detected change point! Halting.")
                    new_halt = self.halted + 1
                    self.halted.set(new_halt)
                    break
            else:
                # otherwise reset rejection count
                rejection_count = 0

        return step * self._sample_rate

    def __detect(self):
        """Step 3 of CPD baseline algorithm
        Forecast AR(p) model and run ks test
        """
        if self._dist_channels is None or self._arp_channels is None:
            raise ValueError("Must run __warmup() first!")

        # begin looping immediatley after warm-up step
        step = self.n0
        chans = self.data.columns

        # run channels in parallel
        tasks = [delayed(self.__detect_chann)(ch, step) for ch in chans]
        results = compute(*tasks)

        # return last-recorded halt-point
        return max(results)

    def __decide(self, count):
        """Step 4 of CPD baseline algorithm
        """
        return count > self.M

    def run(self):
        """Run online CPD algo

        Returns:
            None
        """
        sample = self.__detect()

        # TODO: pickle all objects after running
        return sample

    def find_params(self, iter=50):
        """Find best hyperparameters via random grid-search.

        Args:
            n: Number of iterations

        Modifies:
            n0
            M
            k

        Returns:
            None
        """
        # TODO: how can this be tuned???
        n0 = []


class Offline:
    """Hidden markov process
    """
