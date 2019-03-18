#!/usr/bin/env python3.6

'''
This script standardizes the data for machine learning.
'''

import numpy as np

import itertools
import json
import sys


def load(train, test):
    '''
    Load a train and test data set and export them.

    inputs:
        train = The path to the training set
        test = The path to the testing set

    outputs:
        X_train = The training features
        y_train = The training target feature
        X_test = The test features
        y_test = The test target feature
        meta = The meta data
    '''

    # Open the training and test data
    train = open(train)
    test = open(test)

    # Load the data in opened files
    train = json.load(train)
    test = json.load(test)

    # The target feature is the last feature named 'label'
    X_train, y_train = datasplitter(train['data'])
    X_test, y_test = datasplitter(test['data'])

    # The metadata
    meta = train['metadata']['features']

    return X_train, y_train, X_test, y_test, meta


def datasplitter(x):
    '''
    Split features and target for a dataset.

    inputs:
        x = The data containing features with the last column being the class

    outputs:
        data = The feature data
        target = The target data
    '''

    array = np.array(x, dtype=object)
    data = array[:, :-1]
    target = array[:, -1]

    return data, target


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
    std = np.std(x, ddof=0)

    # In the case that all values are the same
    if std == 0:
        std = 1

    norm = (x-mean)/std

    return norm, mean, std


def trainnorm(x, meta):
    '''
    Standardize the data per column.

    inputs:
        x = The data to be standardized
        meta = The metadata

    outputs:
        xnorm = The standardized data
    '''

    columns = []

    std = []
    mean = []
    count = 0
    for column in x.T:

        # If the column is numeric
        if meta[count][-1] == 'numeric':
            stand = standardize(column)
            std.append(stand[2])
            mean.append(stand[1])
            columns.append(stand[0])

        # If the column is categorical
        else:
            std.append('NaN')
            mean.append('NaN')
            columns.append(column)

        count += 1

    columns = np.array(columns).T

    return columns, mean, std


def testnorm(x, mean, std, meta):
    '''
    Standardize the test data per column based on the train data
    standardization.

    inputs:
        x = The test data to be standardized
        mean = The feature means from traning data
        std = The standard deviation from features of training data
        meta = The metadata

    outputs:
        xnorm = The standardized test data
    '''

    count = 0
    xnorm = []
    for column in x.T:

        # If the column is numerical
        if meta[count][-1] == 'numeric':
            col = []
            for item in column:
                if std[count] == 0:
                    std[count] = 1

                val = (item-mean[count])/std[count]
                col.append(val)

            xnorm.append(col)

        # If the column is categorical
        else:
            xnorm.append(column)

        count += 1

    xnorm = np.array(xnorm).T

    return xnorm


def one_hot_encoding(x, meta):
    '''
    Use one-hot encoding for categorial attributes.

    inputs:
        x = The dataset
        meta = The metadata
    outputs:
        encoded = The dataset with one-hot encoding
    '''

    count = 0
    encoded = []
    for col in x.T:
        if meta[count][-1] == 'numeric':
            encoded.append(col)
        else:

            codedcol = []
            for item in col:
                coded = np.zeros(len(meta[count][-1]), dtype=int)
                coded[meta[count][-1].index(item)] = 1
                codedcol.append(coded)

            for item in np.array(codedcol).T:
                encoded.append(item)

        count += 1

    return np.array(encoded).T


def binarytarget(y, meta):
    '''
    Convert the target data into a binary form. 0 for the first class and
    1 for the second.

    inputs:
        y = The target feature with only two classes

    outputs:
        ybinary = The binary representation of y
    '''

    ybinary = []
    for item in y:
        if item == meta[-1][-1][0]:
            ybinary.append(0)
        if item == meta[-1][-1][1]:
            ybinary.append(1)

    return np.array(ybinary)


def epoch(x, y, w, rate):
    '''
    Calculate the sigmoid output units.

    inputs:
        x = The feature values
        y = The target feature
        w = The weights for the features

    outputs:
        w = The updated weights
        sumerrors = The cross-entropy error sum
        ncorrect = The number of correct predictions
        nincorrect = The number of incorrect precitions
    '''

    sumerrors = 0.0
    prediction = []
    for xrow, yrow in zip(x, y):
        net = np.sum(xrow*w)  # Sum of feature times weight
        o = 1./(1.+np.exp(-net))  # Sigmoid output
        error = -yrow*np.log(o)-(1.-yrow)*np.log(1.-o)  # Cross-entropy
        sub = o-yrow
        gradient = sub*xrow  # Calculate the error gradient
        deltaw = -rate*gradient
        w = w+deltaw  # Update weights

        if o < 0.5:
            prediction.append(0)

        else:
            prediction.append(1)

        sumerrors += error

    prediction = np.array(prediction)
    ncorrect = len(y[np.where(prediction == y)])
    nincorrect = len(y[np.where(prediction != y)])

    return w, sumerrors, ncorrect, nincorrect


def online(x, y, w, rate, epochs):
    '''
    Update weights for a number of epochs.

    inputs:
        x = The feature values
        y = The target feature
        w = The weights for the features

    outputs:
        weights = The updated weigths after n epochs
        errors = The cross-entropy error
        ncorrect = The number of correct predictions
        nincorrect = The number of incorrect precitions
    '''

    weights = []
    errors = []
    ncorrect = []
    nincorrect = []
    for i in epochs:
        w, error, nc, ni = epoch(x, y, w, rate)

        weights.append(w)
        errors.append(error)
        ncorrect.append(nc)
        nincorrect.append(ni)

    return weights, errors, ncorrect, nincorrect


def predict(x, y, w, threshold):
    '''
    Predict the binary class based.

    inputs:
        x = The test set features
        y = The test set target feature
        w = The weights for the features
        threshold = The sigmoid threshold for binary classification

    outputs:
        result = The predictions for logistic regression
        activations = The activation from the sigmoid
    '''

    result = []
    activations = []
    for xrow in x:
        net = np.sum(xrow*w)  # Sum of feature times weight
        o = 1./(1.+np.exp(-net))  # Sigmoid output

        activations.append(o)
        if o < threshold:
            result.append(0)

        else:
            result.append(1)

    result = np.array(result)
    ncorrect = len(y[np.where(result == y)])
    nincorrect = len(y[np.where(result != y)])

    return result, activations, ncorrect, nincorrect


def f1_score(predictions, test):
    '''
    Calculate the F1 score.

    inputs:
        predictions = The predictedions
        test = The true values for the target feature

    outputs:
        f1 = The F1 score
    '''

    tp = len(test[np.where(test == 1)])
    fn = len(test[np.where((test == 1) & (predictions == 0))])
    fp = len(test[np.where((test == 0) & (predictions == 1))])

    recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    f1 = 2.0*precision*recall/(precision+recall)

    return f1


def lr_print(
             epochs,
             errors,
             ncorrect,
             nincorrect,
             activations,
             result,
             test,
             testcorrect,
             testincorrect,
             f1
             ):
    '''
    Print the standard data of logistic regression.

    inputs:
        epochs = The number of epochs for training
        errors = The cross-entropy error
        ncorrect = The number of correct predictions in training
        nincorrect = The number of incorrect predictions in training
        activations = The activations from the sigmoid function
        test = The actual values of the target feature from the test set
        testcorrect = The number of correct predictions
        testincorrect = The number of incorrect predictions
        f1 = The F1 score

    outputs:
        print information to the screen
    '''

    for i, j, k, l in zip(epochs, errors, ncorrect, nincorrect):
        j = '{:.12f}'.format(np.round(j, 12))
        out = ' '.join(map(str, (i, j, k, l)))
        print(out, file=sys.stdout)

    for i, j, k in zip(activations, result, test):
        i = '{:.12f}'.format(np.round(i, 12))
        out = ' '.join(map(str, (i, j, k)))
        print(out, file=sys.stdout)

    out = ' '.join(map(str, (testcorrect, testincorrect)))
    print(out, file=sys.stdout)

    f1 = '{:.12f}'.format(np.round(f1, 12))
    print(f1, file=sys.stdout)
