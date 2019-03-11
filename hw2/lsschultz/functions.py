'''
This script standardizes the data for machine learning.
'''

from scipy import sparse as sparse
import scipy.sparse as sparse
import pandas as pd
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


def mutualinfo(
               col1,
               col1types,
               col2,
               col2types,
               y_train,
               classes,
               n,
               n_classes
               ):
    '''
    Compute the conditional mutual information with binary classes.

    inputs:
        col1 = A feature column
        col1types = The available types for col1
        col2 = A feature column
        col2types = The available types for col2
        y_train = The class data
        classes = The available classes
        n = The number of training instances
        n_classes = The number of classes
    outputs:
        val = The conditional mutual inforamtion between col1 and col2
    '''

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

                # Compute probabilities
                pxixjy = (xixjycount)/(n+n_col1types*n_col2types*n_classes)
                pxixjgiveny = (xixjycount)/(y+n_col1types*n_col2types)
                pxigiveny = (xigivenycount)/(y+n_col1types)
                pxjgiveny = (xjgivenycount)/(y+n_col2types)

                val += pxixjy*np.log2(pxixjgiveny/(pxigiveny*pxjgiveny))

    return val


def mstprim(weights, features, classname, nfeatures):
    '''
    Apply Prim's alogorithm to find MST for an edge list.

    inputs:
        weights = The weights for a graph
        features = The features of interest
        nfeatures = The number of features
    outputs:
        data = Each featres with a list of parents
    '''

    weights = pd.DataFrame(weights, features, columns=features)
    nodes = np.array(features)

    # Apply Prim's Algorithm
    vnew = [nodes[0]]
    enew = []

    while len(vnew) != nfeatures:

        df = weights.loc[vnew, :]
        df = df.drop(vnew, axis=1)

        # Find the edge with the maximum weight
        index = np.where(df.values == np.max(df.values))

        e = (df.iloc[index].index[0], df.iloc[index].columns[0])
        u, v = e

        vnew.append(v)
        enew.append(e)

    data = {}
    for feature in features:
        data[feature] = []
        for e in enew:
            if e[1] == feature:
                data[feature].append(e[0])

        data[feature].append(classname)

    return data


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

    types = [i[-1] for i in meta[:-1]]  # The types of features per column
    classes = meta[-1][-1]  # The available classes
    classname = meta[-1][0]  # The name of the class
    features = [i[0] for i in meta[:-1]]
    nfeatures = X_train.shape[1]  # The number of features
    n = len(y_train)  # The number of instances
    nclasses = len(classes)  # The number of classes

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
                                      classes,
                                      n,
                                      nclasses
                                      )

    # Use Prim's algorithm for MST
    mst = mstprim(mutual, features, classname, nfeatures)

    train = np.column_stack((X_train, y_train))
    features.append(classname)
    types.append(classes)

    typeslength = {i[0]: len(i[1]) for i in meta}

    mstprobs = {}
    for child, parents in mst.items():
        mstprobs[child] = {}

        columns = [child]+parents
        indexcolumns = [features.index(i) for i in columns]

        # Truncate the data to relevant sections
        data = train[:, indexcolumns]

         # The lengths for the features used
        lengths = typeslength[child]

        # Determine the possible combinations of feature items
        a = [types[features.index(i)] for i in columns]
        a = list(itertools.product(*a))
        for i in a:

            count = 0
            conditions = []
            for j in i:
                condition = (data[:, count] == j)
                conditions.append(condition)

                count += 1

            numcondition = conditions[0]
            for j in conditions[1:]:
                numcondition = numcondition & j

            dencondition = conditions[1]
            for j in conditions[2:]:
                dencondition = dencondition & j


            if len(data[numcondition]) == 0:
                num = 0
            else:
                num = len(data[numcondition])

            if len(data[dencondition]) == 0:
                den = 0
            else:
                den = len(data[dencondition])

            num += 1
            den += lengths

            mstprobs[child][i] = num/den

    # Make a table with all the probabilities
    test = {}
    testclasses = {}
    for item in classes:
        items = np.repeat(item, len(y_test))
        test[item] = np.column_stack((X_test, items))
        testclasses[item] = np.column_stack((X_test, items))
    
    for item in classes:
        for child, parents in mst.items():
            columns = [child]+parents
            indexcolumns = [features.index(i) for i in columns]

            # Truncate the data to relevant sections
            data = test[item][:, indexcolumns]

            columnprobs = []
            values = mstprobs[child]
            for row in testclasses[item][:, indexcolumns]:
                for condition, prob in values.items():
                    if condition == tuple(row):
                        columnprobs.append(prob)

            test[item][:, features.index(child)] = columnprobs

    # Multiply the rows together for each of the classes
    rowprods = {}
    for item in classes:

        data = test[item][(test[item][:, -1] == item)]
        data = data[:, :-1].astype(float)
        rowprods[item] = np.prod(data[:, :-1], axis=1)

    # The prior ocunts for each class
    priorcounts = {}
    for item in classes:
        priorcounts[item] = len(y_train[np.where(y_train == item)])

    # The prior probabilities for each class
    priorprobs = {}
    for key, value in priorcounts.items():
        priorprobs[key] = (value+1)/(n+nclasses)

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

    structure = []
    for i, j in mst.items():
        j = [i]+j
        structure.append((tuple(j)))

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
