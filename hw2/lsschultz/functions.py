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

    # The possible classes for targets
    labels = train['metadata']['features'][-1][-1]

    # Data Frames
    dftrain = pd.DataFrame(X_train, columns=features_train)
    dftrain[target_train] = y_train

    dftest = pd.DataFrame(X_test, columns=features_test)
    dftest[target_test] = y_test

    return dftrain, dftest, labels


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


def naive_p(px, py):
    '''
    Compute a naive probability.

    inputs:
        px = P(X|y)
        py = P(y)

    outputs:
        probability = P(y|x)
    '''

    probability = np.prod(px)*py

    return probability


def naive_bayes(dftrain, dftest, labels):
    '''
    Calculate probabilities based on Bayes' Rule. This uses the
    Laplace estimate. Binary classification only.

    inputs:
        df = The data frame containg the features and target data

    outputs:
    '''

    # Arbitrary assignment of positive and negative binary classes
    positive = labels[0]
    negative = labels[1]

    df = pd.concat([dftrain, dftest])

    # Determine P(Y=y). This is computing the prior probability.
    ycounts = dftrain.groupby(dftrain.columns[-1])[dftrain.columns[-1]].count()

    probs = []
    counts = []
    for col in dftrain.columns[:-1]:
        count = dftrain[col].value_counts()+1  # Added for laplace estimate
        total = sum([i+1 for i in count])  # Added for laplace estimate
        px = count/total
        probs.append(px)
        counts.append(count)

    probs = pd.DataFrame(probs).transpose()
    featurecounts = pd.DataFrame(counts).transpose()

    total = sum([ycounts[positive], ycounts[negative]])  # Total number
    yprob = ycounts/total  # The probabilites for each class

    yprob_pos = yprob[positive]  # Positive posterior probability
    yprob_neg = yprob[negative]  # Negative posterior probability

    # Determine P(X_i=x|Y=y) and P(X_i=x|Y!=y) for each value of X_i
    dfcounts = []
    for col in dftrain.columns[:-1]:
        counts = dftrain.groupby(dftrain.columns[-1])[col].value_counts()
        counts /= ycounts#sum([i+1 for i in featurecounts[col]])
        dfcounts.append(counts)

    dfcounts = pd.DataFrame(dfcounts).transpose()

    dfpos = {}
    dfneg = {}
    for col in dftest.columns[:-1]:
        dfpos[col] = dftest[col].replace(dfcounts[col][positive])
        dfneg[col] = dftest[col].replace(dfcounts[col][negative])

    dfpos = pd.DataFrame(dfpos)
    dfneg = pd.DataFrame(dfneg)

    posprod = np.prod(dfpos, axis=1)
    negprod = np.prod(dfneg, axis=1)

    numerator = posprod*yprob_pos
    denominator = yprob_pos*posprod+yprob_neg*negprod

    probabilities = numerator/denominator
    print(probabilities)


def conditional_mutual_information(X, y):
    '''
    Compute the conditional mutual information.

    inputs:
        X = The set of features
        y = The set of targets
    outputs:
        
    '''

    
