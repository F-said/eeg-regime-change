import numpy as np

SIGMA = 5


class Online:
    """
    Benchmark change-point-detection algorithm
    """

    def __init__(self, trained_model, k, verbose=True):
        """
        Args:
            trained_model   : object containing warmed models
            k               : halts when k channels are rejected
            verbose         : option to print
        """
        self.trained_model = self.make_model(trained_model)
        self.verbose = verbose
        self.k = k

        self.forecasts = self.trained_model.forecasts
        self.forecast_window = self.trained_model.forecast_window

    def make_model(self, trained_model):
        if trained_model is None:
            print("No trained model! Please warmup before running.")
        if not trained_model.trained:
            print("Model not trained! Training now.")
            trained_model.warmup()
        return trained_model

    def __update_forecasts(self):
        model = self.trained_model
        sample_rate = model.sample_rate
        arp_channels = model.arp_channels
        ch_names = list(model.data.keys())

        self.forecast_window += self.forecast_window
        for ch in ch_names:
            inc = arp_channels[ch].forecast(self.forecast_window * sample_rate)
            self.forecasts[ch] = inc

    def __predict(self, ch, step):
        """Step 2 of CPD baseline algorithm
        Predicts data in next window of size s

        Returns:
            innovations : prediction error on next window
        """
        # extract all model parameters
        model = self.trained_model
        sample_rate = model.sample_rate
        n0 = model.n0
        data = model.data

        # calculate next window of data (+1 second)
        window = step * sample_rate
        # exit if beyond samples
        if window > data[ch].shape[0]:
            return None

        # predict data in next sample rate window using AR(p) results
        next_forecast = (step - n0) + 1
        # check if we need to make more forecasts
        if next_forecast > self.forecast_window:
            # update forecasts across channels
            self.__update_forecasts()

        pred = self.forecasts[ch]
        stop = next_forecast * sample_rate
        start = stop - sample_rate
        pred = pred[start:stop]

        # calculate innovations in this window by comparing actual data
        next_data = data[ch][window:window + sample_rate]
        innovations = next_data - pred

        return innovations

    def __detect(self):
        """Step 3 of CPD baseline algorithm
        Forecast AR(p) model and run ks test
        """
        # extract all model parameters
        model = self.trained_model
        n0 = model.n0
        data = model.data
        dist_channels = model.dist_channels
        sample_rate = model.sample_rate

        # begin looping immediatley after warm-up step
        step = n0

        # track halted channels
        halted = []
        active = list(data.keys())

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
                if len(halted) >= self.k:
                    if self.verbose:
                        print(f"{self.k} channels halted. Halting detection.")
                    return step * sample_rate

                next_innov = self.__predict(ch, step)
                if next_innov is None:
                    if self.verbose:
                        print(f"No more data! Stopped at step {step}")
                        step = 0
                    return step * sample_rate

                # calc 5-sig
                mu, sig = dist_channels[ch]
                x = np.mean(next_innov)
                z_score = abs((x - mu)/sig)

                # check for deviation greater than N-sigma
                if z_score >= SIGMA:
                    rejected[ch] += 1
                    if self.verbose:
                        print(f"{ch} rejection incremented {rejected[ch]}")
                    # make decision if we should halt
                    if self.__decide(rejected[ch]):
                        if self.verbose:
                            print(f"{ch} detected change point! Halting.")
                        halted.append(ch)
                else:
                    # otherwise reset rejection count
                    rejected[ch] = 0
                    if self.verbose:
                        print(f"{ch} reset rejection count {rejected[ch]}")
            step += 1

        # return last-recorded halt-point
        return step * sample_rate

    def __decide(self, count):
        """Step 4 of CPD baseline algorithm
        """
        M = self.trained_model.M
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
