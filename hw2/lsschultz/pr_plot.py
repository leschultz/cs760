#!/usr/bin/env python3.6

from matplotlib import pyplot as pl

from functions import *

import numpy as np
import argparse

parserdescription = 'Arguments for k, training set, and test set'

parser = argparse.ArgumentParser(
                                 description=parserdescription
                                 )

parser.add_argument(
                    'train',
                    type=str,
                    help='Path to training set'
                    )

parser.add_argument(
                    'test',
                    type=str,
                    help='Path to test set'
                    )

args = parser.parse_args()

# Load the training and testing data
X_train, y_train, X_test, y_test, meta = load(args.train, args.test)

positive = meta[-1][1][0]
negative = meta[-1][1][1]

# For naive
probabilities = naive_bayes(X_train, y_train, X_test, y_test, meta)

# Print the data in standard output
predicted = []
acutal = []
confidence = []
section = zip(
              probabilities['predictions'],
              probabilities['actual'],
              probabilities['probabilities']
              )

for i, j, k in section:
    predicted.append(i)
    acutal.append(j)
    confidence.append(k)

coordinates = pr(y_test, confidence, positive, predicted)
coordinates = np.array(coordinates)

x = coordinates[:, 0]
y = coordinates[:, 1]

fig, ax = pl.subplots()

ax.plot(x, y, label='naive')

ax.set_xlabel('Precision')
ax.set_ylabel('TPR')
ax.grid()
ax.legend(loc='lower right')

fig.tight_layout()
pl.savefig('naive_Precission_Recall.pdf')

# For tan
probabilities = tan(X_train, y_train, X_test, y_test, meta)

# Print the data in standard output
predicted = []
acutal = []
confidence = []
section = zip(
              probabilities['predictions'],
              probabilities['actual'],
              probabilities['probabilities']
              )

for i, j, k in section:
    predicted.append(i)
    acutal.append(j)
    confidence.append(k)

coordinates = pr(y_test, confidence, positive, predicted)
coordinates = np.array(coordinates)

x = coordinates[:, 0]
y = coordinates[:, 1]

fig, ax = pl.subplots()

ax.plot(x, y, label='TAN')

ax.set_xlabel('Precision')
ax.set_ylabel('TPR')
ax.grid()
ax.legend(loc='lower right')

fig.tight_layout()
pl.savefig('TAN_Precission_Recall.pdf')

