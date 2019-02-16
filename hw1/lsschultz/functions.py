'''
This script standardizes the data for machine learning.
'''

from collections import Counter

import numpy as np


def datasplitter(x):
    '''
    Split data and classes for a dataset.

    inputs:
        x = The data containing features with the last column being the class

    outputs:
        data = The feature data
        class = The class label
    '''

    array = np.array(x)
    data = array[:, :-1]

    classes = []
    for i in x:
        classes.append(i[-1])

    return data, classes


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

    columns = []

    for column in x.T:

        # If the column is numeric
        try:
            stand = standardize(column)
            columns.append(stand)

        # If the column is categorical
        except Exception:
            columns.append(column)

    return np.array(columns).T


def hamming(x, y):
    '''
    Compute the Hamming distance.

    inputs:
        x = A training value
        y = A test value

    outputs:
        returns distance
    '''

    return x!=y


def manhattan(x, y):
    '''
    Compute the Manhattan distance.

    inputs:
        x = A training value
        y = A test value

    outputs:
        returns distance.
    '''

    return abs(x-y)


def knn(x, xc, y, k, datatype):
    '''
    Calculate a distance metric for training and give a prediction.
    Manhattan distance for numeric features.
    Hamming distance for categorical features.

    inputs:
        x = Training data
        xc = Training classes
        y = Test data

    outputs:
        results = Class vote with the predicted value
    '''

    results = []
    for test_instance in y:

        # Compute distances between features
        dist = []
        for train_instance in x:

            # Find the distance based on feature type
            count = 0
            distance = 0
            for i, j in zip(test_instance, train_instance):

                # Numerical data type
                if datatype[count] == 'numeric':
                    distance += manhattan(i, j)

                # Categorical data type
                else:
                    distance += hamming(i, j)

                count += 1

            dist.append(distance)

        # Find the indexes of the smallest k values
        indexes = sorted(range(len(dist)), key=lambda k: dist[k])[:k]
        matches = [xc[i] for i in indexes]  # Get the closest features

        # Create a counter for each matched class
        counts = Counter(matches)  # Keeps ordered keys
        result = max(counts, key=counts.get)

        # The available types of classes
        classtypes = sorted(set([i for i in xc]))

        # Votes in order
        votes = []
        for key in classtypes:
            if key in counts:
                votes.append(counts[key])  # Append existing votes
            else:
                votes.append(0)  # Append zero for non existant votes

        # Store the class vote and prediction
        votes.append(result)
        results.append(votes)

    return results
