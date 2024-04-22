import sys
from hmmlearn import hmm
import numpy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        print(self.model_type)



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

    def _init_C(self, data):
        """
        Computes the base cases of the DP array. Every subsignals cost.
        """
        C = np.ones((self.k, data.shape[1], data.shape[1])) * np.inf

        with ProcessPoolExecutor() as executor:
            futures = {}
            for u in range(data.shape[1]):
                sys.stdout.write(f"Dispatched %: {round(u/data.shape[1] * 100, 2)}\r")
                for v in range(u + 1, data.shape[1]):
                    # Submit each task to the pool
                    future = executor.submit(self._cost, data[:, u:v])
                    futures[future] = (u, v)

            print("Dispatched")

            total = len(futures)
            i = 0
            # As each future completes, record its result
            for future in as_completed(futures):
                i+=1
                sys.stdout.write(f"Computed %: {round(i/total * 100, 5)}\r")

                u, v = futures[future]
                C[0, u, v] = future.result()

            print("Computed")
        return C


    def _fill_dp_array(self, C, data):
        """
        Solves the CPD problem for each intermediate k 
        between 2, ... , self.k.
        """
        print("Fill")

        for k in range(1, self.k):
            for u in range(data.shape[1]):
                for v in range(u+k, data.shape[1]):
                    min_cost = np.inf
                    min_t = -1
                    for t in range(u+k-1,v):
                        cost = C[k-1,u,t] + C[0,t+1,v]
                        if cost < min_cost:
                            min_cost = cost
                            min_t = t
                    C[k,u,v] = min_cost

    def _construct_solution(self, C, data):
        """
        Reconstructs the optimal set 
        T = {t_1, ..., t_k} of change
        points according to the DP array.
        """
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

    def find_change_points_opt(self, data):
        """
        Preforms the Opt algorithm
        for CPD.
        """
        C = self._init_C(data)

        self._fill_dp_array(C, data)

        return self._construct_solution(C, data)

    def _best_split(self, t_start, sub_section):
        """
        Calculates the split of the data with
        the smallest cost.
        """
        min_cost = np.inf
        t_star   = -1
        for t in range(sub_section.shape[1]-1):
            cost = self._cost(sub_section[:, :t]) + self._cost(sub_section[:,t+1:])
            if cost < min_cost:
                min_cost = cost
                t_star = t

        return min_cost, t_start + t_star


    def find_change_points_bin(self, data):
        """
        Finds the approximate solution for the change points
        using the binary segmentation algorithm.
        """
        T = data.shape[1]
        L = [0, T]
        while len(L) < self.k+1:
            L = sorted(L)
            G = [0 for _ in range(len(L)-1)]
            for i in range(len(G)):
                split_cost, best_t = self._best_split(L[i], data[:, L[i]:L[i+1]])
                diff = self._cost(data[:, L[i]:L[i+1]]) - split_cost
                G[i] = (diff, best_t)
            next_t = max(G)[1]
            L.append(next_t)

        return sorted(L)


class basicHMM():
    def __init__(self, k, covariance='full'):
        self.k = k
        self.covariance = covariance


    def _init_model(self):
        k = self.k
        covariance = self.covariance
        self.model = hmm.GaussianHMM(n_components=k, covariance_type=covariance, init_params='e', n_iter=20, verbose=True)
        start_prob =  np.zeros(k)
        start_prob[1] = 1
        trans_mat = np.zeros((k,k))
        for i in range(k-1):
            trans_mat[i,i]   = 0.9
            trans_mat[i,i+1] = 0.1
        trans_mat[k-1,k-1]   = 1
        self.model.start_prob_ = start_prob
        self.model.transmat_   = trans_mat

    def get_change_point(self, states):
        diff_pts = [states[i] - states[i-1] for i in range(1, len(states))]
        cps = []
        for i in range(len(diff_pts)):
            if diff_pts[i] != 0:
                cps.append(i+1)
        return cps

    def fit(self, data):
        models = []
        scores = []
        for _ in range(20):
            try:
                self._init_model()
                self.model.fit(data)
                states = self.model.predict(data)
                models.append(self.model)
                scores.append(self.model.score(data))
            except Exception as e:
                print(e)
                print("FAILED TO CONVERGE")

        model = models[np.argmax(scores)]
        return self.get_change_point(model.predict(data))











