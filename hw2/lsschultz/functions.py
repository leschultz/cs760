'''
This script standardizes the data for machine learning.
'''

import numpy as np

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

    # The target feature is the lasat feature named 'label'
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

    array = np.array(x)
    data = array[:, :-1]
    target = np.array([i[-1] for i in x])

    return data, target


def naive_bayes(X_train, y_train, X_test, y_test, meta):
    '''
    Calculate probabilities based on Bayes' Rule. This uses the
    Laplace estimate. Binary classification only.

    inputs:
        df = The data frame containg the features and target data

    outputs:
    '''

    n = len(y_train)  # The number of prior entries

    types = [i[-1] for i in meta]  # The types of features
    classes = meta[-1][-1]  # The available classes
    nclasses = len(classes)  # The number of classes

    # The tree used for probabilities
    head = meta[-1][0]
    nodes = [i[0] for i in meta[:-1]]
    structure = [(i, head) for i in nodes]

    # Compute prior counts for target
    priorcounts = np.unique(y_train, return_counts=True)
    priorcounts = dict(zip(priorcounts[0], priorcounts[1]))

    # Check to make sure all possible classes are counted
    for i in types[-1]:
        if priorcounts.get(i) is None:
            priorcounts[i] = 0

    # Devide the training data based on the training feature
    classdivisions = {}
    for item in classes:
        division = X_train[np.where(y_train==item)[0]]
        classdivisions[item] = division

    # Calculate the counts for each column
    featurecounts = {}
    for key, value in classdivisions.items():

        i = 0  # Iterate for columns in counts
        featurecounts[key] = []
        for col in value.T:
            count = np.unique(col, return_counts=True)
            count = dict(zip(count[0], count[1]))

            # Check to make sure all possible values are counted
            for j in types[i]:
                if count.get(j) is None:
                    count[j] = 0

            featurecounts[key].append(count)
            i += 1

    # The prior probabilities for each class
    priorprobs = {}
    for key, value in priorcounts.items():
        priorprobs[key] = (value+1)/(n+nclasses)

    # Compute the probabilities on prior features for P(X|Y)
    featureprobs = {}
    for key, value in featurecounts.items():

        featureprobs[key] = []
        for col in value:
            probs = {}
            for i in col:
                probs[i] = (col[i]+1)/(sum([j+1 for k, j in col.items()]))

            featureprobs[key].append(probs)

    # Apply the probabilities on the test features
    testprobs = {}
    for item in classes:

        i = 0
        testprobs[item] = np.zeros(X_test.shape)
        for col in X_test.transpose():

            j = 0
            for element in col:
                replacement = featureprobs[item][i][element]
                testprobs[item][j, i] = replacement
                j += 1

            i += 1

    # Multiply the rows together for each of the classes
    rowprods = {}
    for item in classes:
        rowprods[item] = np.prod(testprobs[item], axis=1)

    # For every row multiply the prior probability for each class (binary)
    probabilities = {}
    for item in classes:
        numerator = priorprobs[item]*rowprods[item]
        denominator = priorprobs[classes[0]]*rowprods[classes[0]]

        denominator = 0
        for key in classes:
            denominator += priorprobs[key]*rowprods[key]

        probabilities[item] = numerator/denominator


    # Choose by the maximum probability
    maxprobs = probabilities[classes[0]]  # Initial probabilities
    choice = [classes[0] for i in maxprobs]  # Initial choices
    for item in range(len(classes[1:])):

        k = 0
        for i, j in zip(maxprobs, probabilities[classes[item+1]]):
            if j > i:
                maxprobs[k] = j
                choice[k] = classes[item+1]
            k += 1

    # The number of correct predictions
    ncorrect = np.sum(choice == y_test)

    results = {
               'tree': structure,
               'probabilities': maxprobs,
               'predictions': choice,
               'actual': y_test,
               'ncorrect': ncorrect
               }

    return results


def print_info(results):
    '''
    Print the data to a file.
    '''

    # Print the structe of the bayes net
    for i in results['tree']:
        out = ' '.join(map(str, (i)))
        print(out, file=sys.stdout)

    print(file=sys.stdout)

    # Print the data in standard output
    section = zip(
                  results['predictions'],
                  results['actual'],
                  results['probabilities']
                  )

    for i, j, k in section:
        out = ' '.join(map(str, (i, j, k)))
        print(out, file=sys.stdout)

    print(file=sys.stdout)

    # Print the number of correct matches
    print(results['ncorrect'], file=sys.stdout)

    print(file=sys.stdout)
    


def conditional_mutual_information(X, y):
    '''
    Compute the conditional mutual information.

    inputs:
        X = The set of features
        y = The set of targets
    outputs:
        
    '''

    
