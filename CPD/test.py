from CPD import *
import random
import sys
import numpy as np



"""
test_data = [1] * 100 + [2]*100 + [3]*100
test_data = [test_data[i] + random.random()/8 for i in range(len(test_data))]
test_data = np.array(test_data).reshape(1, -1)
alg = Offline(3, model="gauss", p=2)
#alg = basicHMM(3, covariance='full')
out = alg.find_change_points_bin(eeg_data)
print(out)
#print(alg.find_change_points_opt(eeg_data))
#print(alg.find_change_points_bin(eeg_data))
"""

if __name__ == "__main__":
    dset = sys.argv[1]
    approx = sys.argv[2]
    regimes = int(sys.argv[3])
    freq = int(sys.argv[4])
    model = sys.argv[5] #Gauss, AR
    if model == "AR":
        p = int(sys.argv[6])
    data = np.genfromtxt(dset, delimiter=" ").astype(np.float32)
    print(data.shape)
    data = np.array([data[:, i*freq] for i in range(data.shape[1]//freq)]).T
    print(data.shape)

    if model == "AR":
        alg = Offline(3, model=model, p=p)
    else: 
        alg = Offline(3, model=model)

    if approx == "True":
        print(alg.find_change_points_bin(data))
    else: 
        print(alg.find_change_points_opt(data))


