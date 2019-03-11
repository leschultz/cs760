#!/usr/bin/env python3.6

from functions import *

import argparse

parserdescription = 'Arguments for k, training set, and test set'

parser = argparse.ArgumentParser(
                                 description=parserdescription
                                 )

parser.add_argument(
                    'data',
                    type=str,
                    help='Path to data set'
                    )

args = parser.parse_args()

# Open the training and test data
data = open(args.data)
data = json.load(data)

meta = data['metadata']['features']
data = data['data']
np.random.shuffle(data)

# Gather 10 splits
data_splits = np.array_split(data, 10)

naiveaccuracies = []
tanaccuracies = []
count = 0
for split in data_splits:

    train = np.concatenate(np.delete(data_splits, count, axis=0))

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = split[:, :-1]
    y_test = split[:, -1]

    # Naive method
    probabilities = naive_bayes(X_train, y_train, X_test, y_test, meta)

    navpred = probabilities['predictions']
    navacc = len(y_test[np.where(y_test == navpred)])/len(y_test)
    naiveaccuracies.append(navacc)

    # TAN method
    probabilities = tan(X_train, y_train, X_test, y_test, meta)
    tanpred = probabilities['predictions']
    tanacc = len(y_test[np.where(y_test == tanpred)])/len(y_test)
    tanaccuracies.append(tanacc)

    count += 1

deltas = [i-j for i, j in zip(naiveaccuracies, tanaccuracies)]
mean = np.mean(deltas)
sig = np.std(deltas)
tvals = [(i-mean)/(sig/len(deltas)**0.5) for i in deltas]
print(tvals)
for i in tvals:
    print(i)
