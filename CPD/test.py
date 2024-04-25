from CPD import *
import random
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_single_channel(data):
    T = [i for i in range(data.shape[1])]
    fig,axs = plt.subplots(data.shape[0], 1, sharex=True)
    fig.set_size_inches(30.5, 32.5)
    axs.plot(T, data[0])
    plt.show()

def plot_tsa(data):
    T = [i for i in range(data.shape[1])]
    fig,axs = plt.subplots(data.shape[0], 1, sharex=True)
    fig.set_size_inches(30.5, 32.5)
    for c in range(data.shape[0]):
        axs[c].plot(T, data[c])
    plt.show()


def main():
    freq = sys.argv[2]
    dset = sys.argv[1]
    data = np.genfromtxt(dset, delimiter="\t").astype(np.float32)#.reshape(1, -1)
    print(data.shape)

    if freq == 'plot_s':
        plot_single_channel(data)

    if freq == 'plot':
        plot_tsa(data)

    freq = int(freq)


    print(data.shape)
    data = np.array([data[:, i*freq] for i in range(data.shape[1]//freq)]).T
    print(data.shape)
    model = OfflineWrapper(3, model='gauss', p=100)
    cps = model.find_change_points_bin(data)
    for cp in cps:
        print([item*freq for item in cp])
    if (len(cps) > 1):
        changes = model.get_cp_k_votes(cps,3)
        print("FINAL")
        print([item*freq for item in changes])



if __name__ == "__main__":
    main()
