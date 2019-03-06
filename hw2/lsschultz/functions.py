'''
This script standardizes the data for machine learning.
'''

import pandas as pd
import numpy as np

import json


def load(train, test):
    '''
    Load a train and test data set and export them as data frames.

    inputs:
        train = The path to the training set
        test = The path to the testing set

    outputs:
        dftrain = The training data frame
        dftest = The testing data frame
        labels = The possible classes for the target feature
    '''

    # Open the training and test data
    train = open(train)
    test = open(test)

    # Load the data in opened files
    train = json.load(train)
    test = json.load(test)

    # The target feature is the lasat feature named 'label'
    X_train, y_train = datasplitter(train['data'])
    X_test, y_test = datasplitter(test['data'])

    # The data types
    features_train = [i[0] for i in train['metadata']['features'][:-1]]
    features_test = [i[0] for i in test['metadata']['features'][:-1]]

    # The feature types
    target_train = train['metadata']['features'][-1][0]
    target_test = test['metadata']['features'][-1][0]

    # The metadata
    meta = train['metadata']['features']

    # Data Frames
    dftrain = pd.DataFrame(X_train, columns=features_train)
    dftrain[target_train] = y_train

    dftest = pd.DataFrame(X_test, columns=features_test)
    dftest[target_test] = y_test

    return dftrain, dftest, meta


def datasplitter(x):
    '''
    Split features and target for a dataset.

    inputs:
        x = The data containing features with the last column being the class

    outputs:
        data = The feature data
        class = The target data
    '''

    array = np.array(x)
    data = array[:, :-1]
    targets = np.array([i[-1] for i in x])

    return data, targets


def naive_bayes(dftrain, dftest, meta):
    '''
    Calculate probabilities based on Bayes' Rule. This uses the
    Laplace estimate. Binary classification only.

    inputs:
        df = The data frame containg the features and target data

    outputs:
    '''

    train = np.array(dftrain)
    test = np.array(dftest)

    n = len(train)  # The number of prior entries

    # Compute prior probabilities
    prior = np.unique(train[:, -1], return_counts=True)
    prior = dict(zip(prior[0], prior[1]))

    classes = meta[-1][-1]  # The number of available classes

    # Devide the training data based on the training feature
    classdivisions = {}
    for item in classes:
        division = train[np.where(train==item)[0]]
        classdivisions[item] = division

    # Calculate the counts for each column
    featurecounts = {}
    for key, value in classdivisions.items():
        featurecounts[key] = []
        for col in value.T:
            count = np.unique(col, return_counts=True)
            count = dict(zip(count[0], count[1]))
            featurecounts[key].append(count)

    print(featurecounts)

    # The prior probabilities for each class
    priorprobs = {}
    for key, value in prior.items():
        priorprobs[key] = value/n

    types = [i[-1] for i in meta]  # The types of features
    counts = [len(i) for i in types]


def conditional_mutual_information(X, y):
    '''
    Compute the conditional mutual information.

    inputs:
        X = The set of features
        y = The set of targets
    outputs:
        
    '''

    
