'''
This script standardizes the data for machine learning.
'''

from collections import Counter

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

    # In the case that all values are the same
    if std == 0:
        std = 1

    xnorm = [(i-mean)/std for i in x]

    return xnorm


def autonorm(x):
    '''
    Standardize the data per column.

    inputs:
        x = The data to be standardized

    outputs:
        xnorm = The standardized data
    '''

    xnorm = np.array(x)[:, :-1]
    xnorm = np.apply_along_axis(standardize, 0, xnorm)
    xnorm = [list(i) for i in xnorm]

    count = 0
    for i in x:
        xnorm[count].append(i[-1])
        count += 1

    return xnorm


def hamming(x, y):
    '''
    Compute the Hamming distance.

    inputs:
        x = A training set
        y = A test instance

    outputs:
        distance = The distance
    '''

    distance = sum(i!=j for i, j in zip(x, y))

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

    distance = sum(abs(i-j) for i, j in zip(x, y))

    return distance


def knn(x, y, k, datatype):
    '''
    Calculate a distance metric for training and give a prediction.
    Manhattan distance for numeric features.
    Hamming distance for categorical features.

    inputs:
        x = Training data
        y = Test data

    outputs:
        results = Class vote with the predicted value
    '''

    results = []
    for test_instance in y:
        dist = []

        # Compute distances between features
        for train_instance in x:
            if datatype == 'categorical':
                dist.append(hamming(train_instance[:-1], test_instance[:-1]))

            if datatype == 'numeric':
                dist.append(manhattan(train_instance[:-1], test_instance[:-1]))


        # Find the indexes of the smallest k values
        indexes = sorted(range(len(dist)), key=lambda k: dist[k])[:k]
        matches = [x[i] for i in indexes]

        # Append classes in order
        classes = [i[-1] for i in matches]

        # Create a counter for each matched class
        counts = Counter(classes)  # Keeps ordered keys
        result = max(counts, key=counts.get)

        # The available types of classes
        classtypes = sorted(set([i[-1] for i in x]))

        # Votes in order
        votes = []
        for key in classtypes:
            if key in counts:
                votes.append(counts[key])
            else:
                votes.append(0)

        # Store the class vote and prediction
        votes.append(result)
        results.append(votes)

    return results
