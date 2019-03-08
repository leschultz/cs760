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


def traincount(X_train, y_train, types, classes):
    '''
    Count the number of features and classes.

    inputs:
        X_train = The training features
        y_train = The training target feature
        meta = The meta data

    outputs:
        priorcounts = The counts for the classes
        featurecounts = The counts for the features
    '''

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
        division = X_train[np.where(y_train == item)[0]]
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

    return priorcounts, featurecounts


def testprobabilities(X_train, y_train, X_test, types, classes):
    '''
    Calculate probabilities based on Bayes' Rule.

    inputs:
        X_train = The training features
        y_train = The training target feature
        X_test = The test features
        meta = The meta data

    outputs:
        priorprobs = The class probabilities
        testprobs = The probability array for a test input
    '''

    n = len(y_train)  # The number of prior entries

    nclasses = len(classes)  # The number of classes

    priorcounts, featurecounts = traincount(X_train, y_train, types, classes)

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

    return priorprobs, testprobs


def mutualinfo(col1, col1types, col2, col2types, y_train, classes):
    '''
    Compute the conditional mutual information with binary classes.

    inputs:
        col1 = A feature column
        col1types = The available types for col1
        col2 = A feature column
        col2types = The available types for col2
        y_train = The class data
        classes = The available classes
    outputs:
        val = The conditional mutual inforamtion between col1 and col2
    '''

    n_classes = len(classes)
    n = len(y_train)

    val = 0
    for i in col1types:
        n_col1types = len(col1types)
        xicondition = (col1 == i)

        for j in col2types:
            n_col2types = len(col2types)
            xjcondition = (col2 == j)

            for item in classes:
                ycondition = (y_train == item)
                y = len(y_train[ycondition])

                # Conditions for counting
                xigiveny = xicondition & ycondition
                xjgiveny = xjcondition & ycondition
                xixjy = xicondition & xjcondition & ycondition

                # The counts based on conditions
                xixjycount = len(y_train[xixjy])+1
                xigivenycount = len(col1[xigiveny])+1
                xjgivenycount = len(col2[xjgiveny])+1

                # Compute P(Xi,Xj,Y)
                pxixjy = (xixjycount)/(n+n_col1types*n_col2types*n_classes)
                pxixjgiveny = (xixjycount)/(y+n_col1types*n_col2types)
                pxigiveny = (xigivenycount)/(y+n_col1types)
                pxjgiveny = (xjgivenycount)/(y+n_col2types)

                val += pxixjy*np.log2(pxixjgiveny/(pxigiveny*pxjgiveny))

    return val


def naive_bayes(X_train, y_train, X_test, y_test, meta):
    '''
    Calculate probabilities based on Bayes' Rule. This uses the
    Laplace estimate. Binary classification only.

    inputs:
        X_train = The training features
        y_train = The training target feature
        X_test = The test features
        y_test = The test target feature
        meta = The meta data

    outputs:
        results = A dictionary containing all predictions and figures
    '''

    types = [i[-1] for i in meta]  # The types of features
    classes = meta[-1][-1]  # The available classes

    # The tree used for probabilities
    head = meta[-1][0]
    nodes = [i[0] for i in meta[:-1]]
    structure = [(i, head) for i in nodes]

    # Gather the probabilities for a testing instance
    priorprobs, testprobs = testprobabilities(
                                              X_train,
                                              y_train,
                                              X_test,
                                              types,
                                              classes
                                              )

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


def tan(X_train, y_train, X_test, y_test, meta):
    '''
    Use a tree augmented network (TAN) alorithm for prediction.

    inputs:
        X_train = The training features
        y_train = The training target feature
        X_test = The test features
        y_test = The test target feature
        meta = The meta data

    outputs:
        results = A dictionary containing all predictions and figures
    '''

    types = [i[-1] for i in meta]  # The types of features
    classes = meta[-1][-1]  # The available classes
    nfeatures = X_train.shape[1]  # The number of features
    n = len(y_train)  # the number of instances

    # Compute the conditional mutual information
    mutual = np.zeros((nfeatures, nfeatures))  # Zeros matrix for values

    columns = X_train.transpose()
    n_columns = columns.shape[0]

    for i in range(n_columns):
        for j in range(n_columns):
            mutual[i, j] = mutualinfo(
                                      columns[i],
                                      types[i],
                                      columns[j],
                                      types[j],
                                      y_train,
                                      classes
                                      )
    print(mutual)


def print_info(results):
    '''
    Print the standard data to a file.

    inputs:
        results = A dictionary containing all predictions and figures
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
        k = '{:.12f}'.format(np.round(k, 12))
        out = ' '.join(map(str, (i, j, k)))
        print(out, file=sys.stdout)

    print(file=sys.stdout)

    # Print the number of correct matches
    print(results['ncorrect'], file=sys.stdout)

    print(file=sys.stdout)
