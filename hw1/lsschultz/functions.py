'''
This script standardizes the data for machine learning.
'''

import numpy as np


def standardize(x):
    '''
    Use the mean and standard deviation of sample data to standardize
    features.

    inputs:
        x = The data to be standardized

    outputs:
        xnorm = The standardized data
    '''

    mean = np.mean(x)
    std = np.std(x)

    xnorm = [(i-mean)/std for i in x]

    return xnorm


def hamming(x, y):
    '''
    Compute the Hamming distance.

    inputs:
        x = A training set
        y = A test instance
    outputs:
        distance = The distance.
    '''

    distance = sum(1-(i==j) for i, j in zip(x, y))

    return distance


def manhattan(x, y):
    '''
    Compute the Manhattan distance.

    inputs:
        x = A training set
        y = A test instance
    outputs:
        distance = The distance.
    '''

    print('Need to write code')


def nearest(x, y, k, datatype):
    '''
    Calculate a distance metric for training and give a prediction.
    Manhattan distance for numeric features.
    Hamming distance for categorical features.

    inputs:
        x = Training data
        y = Test data
    outputs:
        distances = The calculated distances
    '''

    distances = []  # List to append distances
    if datatype == 'numeric':

        print('Need to write code')

    if datatype == 'categorical':
        for test_instance in y:
            dist = []
            for train_instance in x:
                dist.append(hamming(train_instance, test_instance))

            # Find the indexes of the smallest k values
            indexes = sorted(range(len(dist)), key=lambda k: dist[k])[:k]
            matches = np.array(x)[indexes]

            # Create a counter for each matched class
            classes = [i[-1] for i in matches]
            counts ={}
            for item in set(classes):
                counts[item] = classes.count(item)

            counts['rep'] = 6  # Test code

            # Find the maximum vote
            result = max(counts, key=lambda i: counts[i])
            maxval = counts[result]

            # Break a tie if it exists

            print(high)
            print(result)

            print(classes)
            print(counts)

            distances.append(np.array(dist))

    return distances
