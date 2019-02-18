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


def calculate_distances(x, y, datatype):
    '''
    Compute the distances betwen training and test sets.

    inputs:
        x = The training set
        y = The test set
        datatype = The feature data type values
        
    outputs:
        distances = The distances between training and test sets
    '''

    distances = []
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

        distances.append(dist)

    return distances


def conficence(distances, xc, predictions, k, positive, epsilon):
    '''
    Compute the confidence values.

    inputs:
        distances = The distances between training and test features
        xc = The training classes
        predictions = The predicted classes
        k = The number of nearest neighbors
        positive = The positive class
        epsilon = A parameter for avoiding issues when test and train sets are
                  equal

    outputs:
        confidences = The probability P(y=1|x)
    '''

    confidences = []
    for dist in distances:        

        # Find the indexes of the smallest k values
        indexes = sorted(range(len(dist)), key=lambda k: dist[k])[:k]

        # Get the closest training sets
        matches_classes = [xc[i] for i in sorted(indexes)]
        matches_dists = [dist[i] for i in sorted(indexes)]

        # Compute the weights
        weights = [1/(i**2+epsilon) for i in matches_dists]

        # Compute the conficence values
        num = sum([(i==positive)*j for i, j in zip(matches_classes, weights)])
        den = sum(weights)

        p = num/den
        confidences.append(p)

    return confidences


def roc(yc, confidences, positive):
    '''
    Generate the data for ROC.

    inputs:
        confidences = The confidence values
        yc = The true classes for the test set
        positive = The positive class

    outputs:
        coordinates = The (FPR, TPR) coordinates
    '''

    # Sort the data based on predicted confidence
    rocdata = [(i, j) for i, j in zip (confidences, yc)]
    rocdata = sorted(rocdata, reverse=True)

    length = len(rocdata)

    # The number of negative/positve instances in the test set
    num_neg = sum((i!=positive) for i in yc)
    num_pos = length-num_neg

    tp = 0
    fp = 0
    last_tp = 0
    coordinates = []  # The FPR and TPR coordinates (FPR, TPR)
    for i in range(length):

        # List of conditions to find the thresholds where there is a positive
        # on the high side and a negative instance on the low side
        condition = (i > 1)
        condition = condition and (rocdata[i][0]!=rocdata[i-1][0])       
        condition = condition and (rocdata[i][1]!=positive)
        condition = condition and (tp>last_tp)       

        if condition:
            fpr = fp/num_neg
            tpr = (tp)/num_pos
            last_tp = tp

            coordinates.append((fpr, tpr))

        if rocdata[i][1] == positive:
            tp += 1

        else:
            fp += 1

    fpr = fp/num_neg
    tpr = tp/num_pos

    coordinates.append((fpr, tpr))

    return coordinates


def prediction_accuracy(x, y):
    '''
    Compute the accuracy of predictions.

    inputs:
        x = Predicted values
        y = Actual values

    outputs:
        accuracy = The accuracy of predictions
    '''

    n = len(x)  # Determine length from the predicted values
    accuracy = sum(x==y)/n  # The sum of true values

    return accuracy


def knn(xc, distances, k, labels):
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
    for dist in distances:

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
             train_classes,
             validation_classes,
             distances,
             kmax,
             labels
             ):
    '''
    Tune kNN for hyperparameters.

    inputs:
        train_classes = The classes from the training data
        validation_classes = The classes from the validation data
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
                     train_classes,
                     distances,
                     i,
                     labels,
                     )

        result = np.array(result)[:, -1]

        # Compute the accuracy of predictions
        accuracy = prediction_accuracy(result, validation_classes)

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
