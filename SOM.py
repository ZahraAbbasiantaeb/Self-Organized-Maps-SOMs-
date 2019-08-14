import math
import pickle
import random
from random import randrange

import numpy
import numpy as np
from sklearn.metrics import accuracy_score

from data import  plot_predicted_clusters, plot_wins
from data import data, tags

map_size = 10
dim = 2
neighbor_count = 4
neighbors =  [[[-1 for k in range(neighbor_count)] for j in range(map_size)] for i in range(map_size)]
wins = np.zeros(map_size * map_size)



def create_codebooks(data):

    codebooks =[]
    for i in range(0, map_size):
        for j in range(0, map_size):
            tmp = randrange(len(data))
            codebooks.append(data[tmp])
    
    with open('codebook.pkl' , 'wb') as output:
        pickle.dump(codebooks, output)
    return

def init_codeBooks():

    with open('codebook.pkl', 'rb') as input:
        codebooks = pickle.load(input)

    return codebooks


def define_neighbors(type='rectangular'):

    if(type=='rectangular'):
        for i in range(0, map_size):

            for j in range(0, map_size):

                tmp = 0

                if(i-1>=0):
                    neighbors[i][j][tmp] = (i-1)*map_size+j
                    tmp+=1

                if(i+1<map_size):
                    neighbors[i][j][tmp] = (i+1)*map_size+j
                    tmp+=1

                if(j-1>=0):
                    neighbors[i][j][tmp] = i*map_size+j-1
                    tmp+=1

                if(j+1<map_size):
                    neighbors[i][j][tmp] = i*map_size+j+1
                    tmp+=1


    elif (type == 'hexagonal'):

        for i in range(0, map_size):

            for j in range(0, map_size):

                tmp = 0

                if(i-1 >= 0):
                    neighbors[i][j][tmp] = (i - 1) * map_size + j
                    tmp += 1

                    if(j-1 >= 0):
                        neighbors[i][j][tmp] = (i - 1) * map_size + j-1
                        tmp += 1

                if(i+1 < map_size):
                    neighbors[i][j][tmp] = (i + 1) * map_size + j
                    tmp += 1

                    if (j + 1 < map_size):
                        neighbors[i][j][tmp] = (i + 1) * map_size + j + 1
                        tmp += 1

                if(j-1 >= 0):
                    neighbors[i][j][tmp] = i * map_size + j - 1
                    tmp += 1

                if(j+1 < map_size):
                    neighbors[i][j][tmp] = i * map_size + j + 1
                    tmp += 1


    elif (type == 'circular'):

        for i in range(0, map_size):

            for j in range(0, map_size):

                tmp = 0

                if (i - 1 >= 0):
                    neighbors[i][j][tmp] = (i - 1) * map_size + j
                    tmp += 1

                    if (j - 1 >= 0):
                        neighbors[i][j][tmp] = (i - 1) * map_size + j - 1
                        tmp += 1

                    if (j + 1 < map_size):
                        neighbors[i][j][tmp] = (i - 1) * map_size + j + 1
                        tmp += 1

                if (i + 1 < map_size):
                    neighbors[i][j][tmp] = (i + 1) * map_size + j
                    tmp += 1

                    if (j + 1 < map_size):
                        neighbors[i][j][tmp] = (i + 1) * map_size + j + 1
                        tmp += 1

                    if (j - 1 >=0):
                        neighbors[i][j][tmp] = (i + 1) * map_size + j - 1
                        tmp += 1

                if (j - 1 >= 0):
                    neighbors[i][j][tmp] = i * map_size + j - 1
                    tmp += 1

                if (j + 1 < map_size):
                    neighbors[i][j][tmp] = i * map_size + j + 1
                    tmp += 1



    return


def euclidean_dist(x1, x2):

    return numpy.linalg.norm(x1-x2)


def get_winner(input):

    index = 0
    dist = euclidean_dist(codebooks[0], input)

    for i in range(1, len(codebooks)):
        tmp = euclidean_dist(codebooks[i], input)

        if(tmp<dist):
            dist = tmp
            index = i

    return index


def update_weight(winner, neighbor, input, sigma, Beta):

    coef = (Beta * math.exp(-euclidean_dist(winner, neighbor)/(2*sigma**2)))
    return  coef * (input - neighbor)


def update_winner(winner, input, Beta):

    return  Beta * (input - winner)


def find_nearest(centers):

    first = 0
    second = 1
    dist = euclidean_dist(centers[0],centers[1])

    for i in range(0, len(centers)):

        for j in range(i+1, len(centers)):

            tmp = euclidean_dist(centers[i],centers[j])

            if(tmp<dist):

                dist = tmp
                first = i
                second = j

    return first, second


def find_clusters(codebooks, cluster_count):

    centers = codebooks


    while(len(centers)>cluster_count):

        i,j = find_nearest(centers)
        vector = (centers[i]+centers[j])/2
        new_centers = []

        for row in range(0, len(centers)):
            if(row!=i and row!=j):
                new_centers.append(centers[row])
        new_centers.append(vector)
        centers = new_centers


    return centers


def SOM(data, radius, Beta_0):

    T = len(data)

    sigma = radius

    Beta = Beta_0

    for i in range(0, T):

        input = data[i]

        winner_index = get_winner(input)

        winner = codebooks[winner_index]

        wins[winner_index] += 1

        row = int(winner_index/map_size)
        col = int(winner_index%map_size)

        for elem in neighbors[row][col]:

            if(elem == -1):
                break

            codebooks[elem] += update_weight(winner, codebooks[elem], input, sigma, Beta)

        codebooks[winner_index] += update_winner(winner, input, Beta)

        sigma = radius*(1-(i/T))
        Beta = Beta_0*(1-(i/T))

    return codebooks, wins


def SOM_without_border(data, Beta_0):

    T = len(data)

    Beta = Beta_0

    for i in range(0, T):

        input = data[i]

        winner_index = get_winner(input)

        winner = codebooks[winner_index]

        wins[winner_index] += 1

        for i in range(0, len(codebooks)):
            if(i!=winner_index):
                codebooks[i] += update_weight(winner, codebooks[i], input, 1, Beta)

        codebooks[winner_index] += update_winner(winner, input, Beta)

        Beta = Beta_0*(1-(i/T))

    return codebooks, wins


def show_centers(centers, predicted_tags):

    x = []
    y = []

    for elem in centers:
        x.append(elem[0])
        y.append(elem[1])

    z = [9 for i in range(0, len(centers))]
    plot_predicted_clusters(x, y, z, data, predicted_tags)

    return


def nearest_center(elem, cluster_centers):

    dist = euclidean_dist(elem, cluster_centers[0])
    index = 0

    for i in range(1, len(cluster_centers)):
        tmp = euclidean_dist(elem, cluster_centers[i])

        if(tmp<dist):
            dist = tmp
            index = i

    return index


def assign_clusters(data, cluster_centers):

    tag=[]

    for elem in data:
        tag.append(nearest_center(elem, cluster_centers))

    return tag


def purity_score(y_true, y_pred):

    y_voted_labels = np.zeros(len(y_true))
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])

    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]


    labels = np.unique(y_true)

    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def reduce_dimension(data, centers):

    new_data = np.zeros((len(data),2))
    index = 0

    for elem in data:
        tag = nearest_center(elem, centers)
        new_data[index, 0] = int(tag/map_size)
        new_data[index, 1] = int(tag % map_size)
        index += 1

    return new_data

codebooks = init_codeBooks(data)


define_neighbors()


codebooks, wins = SOM_without_border(data,  1)

centers = find_clusters(codebooks, 6)


predicted_tags = assign_clusters(data, centers)

show_centers(centers, predicted_tags)

print(purity_score(tags, predicted_tags))

# reduced_dimension = reduce_dimension(data, codebooks)
# print(reduced_dimension)

# with open('high_dim_1024_reduced.pkl', 'wb') as output:
#     pickle.dump(reduced_dimension,output)

plot_wins(wins)

