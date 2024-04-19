import numpy as np

MAX_LAGS = 128*(1/2)
SIGMA = 5


class Online:
    """
    Benchmark change-point-detection algorithm
    """

    def __init__(self, trained_model, verbose=True):
        """
        Args:
            trained_model   : object containing warmed models
            verbose : option to print
        """
        self.trained_model = self.make_model(trained_model)
        self.verbose = verbose

    def make_model(self, trained_model):
        if self.trained_model is None:
            print("No trained model! Please warmup before running.")
        if not self.trained_model.trained:
            print("Model not trained! Training now.")
            self.trained_model.warmup()
        return trained_model

    def __predict(self, ch, step):
        """Step 2 of CPD baseline algorithm
        Predicts data in next window of size s

        Returns:
            innovations : prediction error on next window
        """
        # extract all model parameters
        sample_rate = self.trained_model.sample_rate
        fore_window = self.trained_model.forecast_window
        n0 = self.trained_model.n0
        data = self.trained_model.data
        M = self.M
        arp_channels = self.arp_channels
        forecasts = self.forecasts

        # calculate next window of data (+1 second)
        window = step * sample_rate
        # exit if beyond samples
        if window > data.shape[0]:
            return None

        # predict data in next sample rate window using AR(p) results
        next_forecast = (step - n0) + 1
        # check if we need to make more forecasts
        if next_forecast > fore_window:
            fore_window += M
            inc = arp_channels[ch].forecast(fore_window * sample_rate)
            forecasts[ch] = inc

        pred = forecasts[ch]
        print(pred.shape)
        stop = next_forecast * sample_rate
        start = stop - sample_rate
        pred = pred.iloc[start:stop]
        print(pred.shape)

        # calculate innovations in this window by comparing actual data
        next_data = self.data.loc[window:window + (self._sample_rate - 1), ch]
        innovations = next_data - pred
        print(innovations)

        return innovations.values

    def __detect(self):
        """Step 3 of CPD baseline algorithm
        Forecast AR(p) model and run ks test
        """
        # extract all model parameters
        n0 = self.trained_model.n0
        data = self.trained_model.data
        k = self.k
        dist_channels = self.dist_channels

        # begin looping immediatley after warm-up step
        step = n0

        # track halted channels
        halted = []
        active = data.columns

        # track reject count
        rejected = {}
        for ch in active:
            rejected[ch] = 0

        # run channels synchronously
        next_innov = 0

        while next_innov is not None:
            # update active list
            active = list(set(active) - set(halted))

            if self.verbose:
                print(f"STEP {step}")
            for ch in active:
                if len(halted) >= k:
                    if self.verbose:
                        print(f"{k} channels halted. Halting detection.")
                    break

                if self.verbose:
                    print(f"{ch}: ")

                next_innov = self.__predict(ch, step)
                if next_innov is None:
                    if self.verbose:
                        print(f"No more data! Stopped at step {step}")
                        step = 0
                    break

                # calc 5-sig
                mu, sig = dist_channels[ch]
                x = np.mean(next_innov)
                z_score = abs((x - mu)/sig)
                print(z_score)

                # check for deviation greater than N-sigma
                if z_score >= SIGMA:
                    if self.verbose:
                        print(f"{ch} Rejection incremented")
                    rejected[ch] += 1
                    # make decision if we should halt
                    if self.__decide(rejected[ch]):
                        if self.verbose:
                            print(f"{ch} detected change point! Halting.")
                        halted.append(ch)
                        break
                else:
                    if self.verbose:
                        print("Alles ist gut")
                    # otherwise reset rejection count
                    rejected[ch] = 0
            step += 1

        # return last-recorded halt-point
        return step * self._sample_rate

    def __decide(self, count):
        """Step 4 of CPD baseline algorithm
        """
        M = self.M
        return count > M

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
