from CPD import *
import random
import sys
import numpy as np



freq = 100
eeg_data = np.genfromtxt("../data/reduced.txt", delimiter=" ").astype(np.float32)
print(eeg_data.shape)
eeg_data = np.array([eeg_data[:, i*freq] for i in range(eeg_data.shape[1]//freq)]).T
print(eeg_data.shape)
#test_data = np.array(test_data).reshape(1, -1)
alg = Offline(3, model="AR", p=2)
print(alg.find_change_points(eeg_data))



