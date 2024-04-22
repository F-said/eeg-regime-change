import sys
import numpy
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

class Online:
    """Bayesian w/innovation

    Would it be possible to make a change-point
    detection algorithm with NO change-point, and
    rather "train as you go", with some metric for
    outlier detection.

    Well, PCA would definitley help, although it
    would also assume some time for "training"
    but maybe it could be minimal

    The problem with PCA is the fact that it needs
    the entire df to figure 

    Power Spectral Analysis of PCA components
    Needed: Live PCA (X)
    Needed: Live wavelet transform

    1) Check if wavelet transform of PCA is similair to original
    2) Find iterative PCA algo
    3) Find wavelet transform algo
    4) Find good way to combine the two without being too slow
    5) Run & get results
    """
    ...

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
        X = np.zeros((n, self.p+1))
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

