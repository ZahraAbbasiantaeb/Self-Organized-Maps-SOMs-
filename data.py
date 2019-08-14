import pickle
from random import shuffle
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def plot_data(data):

    plt.scatter(data[:,0], data[:,1],  c=tags, alpha=0.5)
    plt.show()

    return


def plot_predicted_clusters(x,y,z, data, tag):

    area = [50 for i in z]
    plt.scatter(x, y, c=z, alpha=0.9, s=area, marker='*')
    plt.scatter(data[:, 0], data[:, 1], c=tag, alpha=0.3)
    plt.show()

    return


def plot_wins(wins):

    x = np.arange(len(wins))
    plt.bar(x, wins)
    plt.show()

    return


def create_data():

    data = loadtxt("dataset/2D.txt", delimiter="\t", unpack=False)
    np.random.shuffle(data)

    with open('data.pkl', 'wb') as output:
        pickle.dump(data, output)

    return



def load_data():
    with open('data.pkl' , 'rb') as input:
        return pickle.load(input)


# create_data()
data = load_data()
tags = data[:,2]
data = data[:,0:2]
# plot_data(data)

