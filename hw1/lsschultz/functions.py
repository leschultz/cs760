'''
This script standardizes the data for machine learning.
'''

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
        mean = The mean of the features
        std = The standard deviation of the features
    '''

    mean = np.mean(x)
    std = np.std(x)

    # In the case that all values are the same
    if std == 0:
        std = 1

    xnorm = [(i-mean)/std for i in x]

    return xnorm, mean, std


def trainnorm(x):
    '''
    Standardize the data per column.

    inputs:
        x = The data to be standardized

    outputs:
        xnorm = The standardized data
    '''

    columns = []

    std = []
    mean = []
    for column in x.T:

        # If the column is numeric
        try:
            stand = standardize(column)
            std.append(stand[2])
            mean.append(stand[1])
            columns.append(stand[0])

        # If the column is categorical
        except Exception:
            std.append('NaN')
            mean.append('NaN')
            columns.append(column)

    return np.array(columns).T, mean, std


def testnorm(x, mean, std):
    '''
    Standardize the test data per column based on the train data
    standardization.

    inputs:
        x = The test data to be standardized
        mean = The feature means from traning data
        std = The standard deviation from features of training data

    outputs:
        xnorm = The standardized test data
    '''

    count = 0
    xnorm = []
    for column in x.T:

        # If the column is numerical
        try:
            col = []
            for item in column:
                if std[count] == 0:
                    std[count] = 1

                val = (item-mean[count])/std[count]
                col.append(val)

            xnorm.append(col)

        # If the column is categorical
        except Exception:
            xnorm.append(column)

        count += 1

    xnorm = np.array(xnorm).T

    return xnorm


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


def knn(x, xc, y, k, datatype, labels):
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
        matches = [xc[i] for i in sorted(indexes)]  # Get the closest features

        # Predict based on order of metadata
        counts = {i: 0 for i in labels}
        for item in matches:
            if item in labels:
                counts[item] += 1

        # Find the maximum counts with the first key with a majority breaking
        # the tie
        result = max(counts, key=counts.get)

        # Votes in order
        votes = []
        for key in counts:
            votes.append(counts[key])

        # Store the class vote and prediction
        votes.append(result)
        results.append(votes)

    return results


def tune_knn(
             train,
             train_classes,
             validation,
             validation_classes,
             kmax,
             types,
             labels
             ):
    '''
    Tune kNN for hyperparameters.

    inputs:
        train = Training data
        train_classes = The classes from the training data
        validation = The validation data
        validation_classes = The classes from the validation data
        types = The data types for the training data
        labels = The possible labels for classes
        kmax = The maximum number of k-nearest neighbors

    outputs:
        accuracies = The accuracy for each k value used
    '''

    # The number of validation instances
    n = len(validation_classes)

    # Convert to array for boolean comparison
    validation_classes = np.array(validation_classes)

    accuracies = {}
    for i in range(1, kmax+1):

        # Use the k-nearest neighbors alorithm
        result = knn(
                     train,
                     train_classes,
                     validation,
                     i,
                     types,
                     labels
                     )

        result = np.array(result)[:, -1]
        accuracy = sum(result == validation_classes)/n  # Sum of true values

        accuracies[i] = accuracy

    return accuracies


def choose_k(accuracies):
    '''
    Choose the k value based on highest accuracy.

    inputs:
        accuracies = The accuracy for each k value used

    outputs:
        k = The optimal k value
    '''

    k = max(accuracies, key=accuracies.get)

    return k
