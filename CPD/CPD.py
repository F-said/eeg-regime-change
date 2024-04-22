import sys
import numpy
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

from sdt.changepoint import BayesOnline
from scipy.signal import butter, filtfilt


class Online:
    """1/3 of "Time-Series Anomaly Detection Service at Microsoft"
    https://arxiv.org/pdf/1906.03821.pdf

    Calculates the spectral residual on a window of "warmup data"
    which is the log spectrum subtracted by the averaged log spectrum.
    This "compressed representation of the sequence while the innovation
    part of the original sequence becomes more significant."

    R(f) = (hq(f) * log(A(f))) - log(A(f))
    A(f) = Amplitude of window of DFT
    P(f) = Phase of window of DFT

    This spectral residual is then transformed back into the spatial
    domain via inverse DFT, which is called the saliency map.

    S(x) = abs(invfft(e^(R(f) + iP(f))))

    Next, the Microsoft people apply a CNN to detect anomolies,
    however, I will instead apply the Bayesian online predictor from
    "Bayesian Online Changepoint detection": https://arxiv.org/pdf/0710.3742.pdf
    on this transformed data.

    For every additional window of data, we apply the saliency map transformation
    & then use the online bayesian change-point detection to check if a regime
    change occured
    TODO: make Online a template class
    """

    def __init__(self, trained_model, k, prob=0.80, verbose=True):
        """
        Args:
            trained_model   : object containing warmed models
            k               : halts when k channels are rejected
            prob            : tolerated probability
            verbose         : option to print
        """
        self.trained_model = self.make_model(trained_model)
        self.verbose = verbose
        self.prob = prob
        self.k = k

        self.bayesian_cpd = self.trained_model.bayesian_cpd
        self.sample_rate = self.trained_model.sample_rate
        # init high-pass filter
        self.b, self.a = butter(
            N=2, Wn=1/(0.5*self.sample_rate),
            btype='high',
            analog=False
        )

    def make_model(self, trained_model):
        if trained_model is None:
            print("No trained model! Please warmup before running.")
        if not trained_model.trained:
            print("Model not trained! Training now.")
            trained_model.warmup()
        return trained_model

    def __saliency(self, ch, step):
        """Calculate saliency map in next window of size s
        for channel

        Returns:
            innovations : prediction error on next window
        """
        # extract model parameters
        model = self.trained_model
        data = model.data

        # calculate next window of data (+1 second)
        window = step * self.sample_rate
        # exit if beyond samples
        if window >= data[ch].shape[0]:
            print("Past!")
            return None
        window_data = data[ch][window:window + self.sample_rate]

        # calc saliency
        # compute high-pass filter
        filt_data = filtfilt(self.b, self.a, window_data)

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

        return saliency

    def __detect(self):
        """Check if next saliency map
        is detected as a change-point via bcpd
        """
        # extract all model parameters
        model = self.trained_model
        n0 = model.n0
        data = model.data
        sample_rate = model.sample_rate

        # begin looping immediatley after warm-up step
        step = n0

        # track halted channels (and when they've halted)
        halted = []
        active = list(data.keys())
        halted_times = {}

        # run channels synchronously
        next_innov = 0

        # kepep running until we run out of data
        while next_innov is not None:
            # update active list
            active = list(set(active) - set(halted))

            if self.verbose:
                print(f"STEP {step}")
            # break out of loop if all channels inactive
            if len(active) == 0:
                break

            # check bayesian prob for each channel
            for ch in active:
                if len(halted) >= self.k:
                    if self.verbose:
                        print(f"{self.k} channels halted. Halting detection.")
                    return halted_times

                next_map = self.__saliency(ch, step)
                if next_map is None:
                    if self.verbose:
                        print(f"No more data! Stopped at step {step}")
                        step = 0
                    return halted_times

                # update bayesian changepoint detector and halt if detected
                # for next saliency
                for i in range(0, len(next_map), sample_rate // 4):
                    try:
                        self.bayesian_cpd[ch].update(next_map[i])
                    except:
                        continue

                    prob = self.bayesian_cpd[ch].get_probabilities(3)
                    if len(prob) >= 1 and np.any(prob[1:] > self.prob):
                        if self.verbose:
                            print(f"{ch} detected change point! Halting.")
                        halted.append(ch)
                        halted_times[ch] = step * sample_rate
                        break

            step += 1

        # return last-recorded halt-point
        return halted_times

    def run(self):
        """Run online CPD algo

        Returns:
            None
        """
        sample = self.__detect()

        # TODO: pickle objects after running
        return sample


class AR():
    """
    Simple AR model. Made to speed up AR computation.
    Assuming this implementation is faster than the
    stat libraries.
    """

    def __init__(self, p):
        self.p = p
        self.params = None

    def _build_lag_matrix(self, data):

        n = data.shape[0]
        nans = np.ones(self.p) * np.nan
        data = np.concatenate((nans, data), axis=0)
        X = np.ones((n, self.p+1))
        for i in range(self.p):
            X[:, i+1] = data[i: -self.p + i]
        return X

    def fit(self, data):
        if len(data.shape) != 1:
            raise Exception ("1d data neccessary")
        X = self._build_lag_matrix(data)[self.p:, :]
        Y = data[self.p:]
        self.params = np.linalg.pinv(X.T @ X) @ X.T @ Y

    def predict(self, data):
        if self.params is None:
            raise Exception ("Did Not Fit Model.")
        X = self._build_lag_matrix(data)
        return X @ self.params
        
        

class Offline:

    """
    Offline Algorithm for change point detection:
        - Cost function -> MSE
        - Internal Model -> AR / Gaussian
        - Search Algorithm -> Opt / Bin Seg
    Description:
        This algorithm is essentially a search algorithm. Searching different
        change points, attempting to find the set of change points which
        minimize some cost function. We include a optimal search which
        returns the optimal solution and a approximate search algorithm.

    Possible Innovations:
        -Weight loss function by channels
            Some channels display change points more
            clearly than others, thus we should make sure the model's
            cost on this channel/channels is low.
        -Learnable weights on the channels loss vs hyper parameter.
    """
    def __init__(self, k : int , model="gauss", p : int = -1):
        """
        Solves the CPD problem for K different
        regimes (K-1) regime changes, Using an
        internal model of AR(p). 
        """
        self.k = k
        self.p = p
        self.model_type = model



    def _fit_model(self, data : np.array):
        """
        data: sub_section of the data which we will fit a model
        to. data is of the form (C, T') where C is the # of channels
        and T' is the # of timesteps in the data.
        """
        if self.model_type == "AR":
            models = []
            for i in range(data.shape[0]):
                m = AR(self.p)
                m.fit(data[i])
                models.append(m)
            return models
        else:
            mu = data.sum(axis=1, keepdims=True) / data.shape[1]
            sigma = 1/(data.shape[1] -1) * (data - mu) @ (data-mu).T

            return mu, sigma

    def _log_liklihood(self, mu, sigma, data):
        """
        Calculates the pseudo log liklihood. We don't include the normalizing
        term at the beginning of the Gaussian PDF. This acts as a metric of
        how well the data fits to the model.
        """
        val = 0
        sigma_inv = np.linalg.pinv(sigma)
        n = data.shape[0]
        liklihood = 0
        mu = mu.reshape(-1)
        for i in range(data.shape[1]):
            y = data[:, i]
            lik =  (-1/2) * ( (y - mu).T @ sigma_inv @ (y-mu) ) 
            liklihood+=lik
            
        return liklihood


    def _cost(self, sub_section : np.array):
        """
        sub_section: sub_section of the data which we will fit a model
        to and calculate the cost for. 
        sub_section is of the form (C, T) where C is the # of channels
        and T is the # of timesteps.
        """
        if self.model_type == "AR":
            if sub_section.shape[1] <= self.p+3:
                return np.inf
            models = self._fit_model(sub_section)
            channel_mse = 0
            for i in range(sub_section.shape[0]):
                model = models[i]
                preds = model.predict(sub_section[i])[self.p:]
                temp = sub_section[i, self.p:]
                channel_mse += ((temp - preds)**2 / (temp.shape[0])).sum()
            channel_mse /= sub_section.shape[0]
            return channel_mse
        else:
            if sub_section.shape[1] <= 1:
                return np.inf

            mu, sigma = self._fit_model(sub_section)
            return - self._log_liklihood(mu, sigma, sub_section)

    def find_change_points(self, data):

        ##Initialize DP array
        C = np.zeros((self.k, data.shape[1], data.shape[1]))
        for u in range(data.shape[1]):
            sys.stdout.write(f"Init %: {round(u/data.shape[1] * 100, 2)}\r")
            for v in range(u+1, data.shape[1]):
                temp = self._cost(data[:,u:v])
                C[0,u,v] = temp

        print("Fill")
        #Fill DP Array
        for k in range(2, self.k-1):
            for u in range(data.shape[1]):
                for v in range(u+k, data.shape[1]):
                    min_cost = np.inf
                    min_t = -1
                    for t in range(u+k-1,v):
                        cost = c[k-1,u,t] + c[0,t+1,v]
                        if cost < min_cost:
                            min_cost = cost
                            min_t = t
                    c[k,u,v] = cost

        print("Reconstruct")
        #Reconstruct Solution
        L = np.zeros(self.k+1)
        L[self.k] = data.shape[1]-1
        k = self.k
        while k > 1:
            s = int(L[k])
            t_star = -1
            min_cost = np.inf
            for t in range(k-1, s):
                cost = C[k-1, 0, t] + C[0, t+1, s]
                if cost < min_cost:
                    min_cost = cost
                    t_star = t
            L[k-1] = t_star
            k -= 1

        return L

