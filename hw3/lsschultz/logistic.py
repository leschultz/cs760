#!/usr/bin/env python3.6

from functions import *

import numpy as np

import argparse

np.random.seed(0)

parserdescription = 'Neural Network with logistic regression'

parser = argparse.ArgumentParser(
                                 description=parserdescription
                                 )
parser.add_argument(
                    'rate',
                    type=float,
                    help='The learning rate'
                    )
parser.add_argument(
                    'epochs',
                    type=int,
                    help='The number of epochs'
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

# The number of rows and columns for training data excluding the classes
rows, cols = X_train.shape

# Normalize the numeric features
X_train_norm, mean, std = trainnorm(X_train, meta)
X_test_norm = testnorm(X_test, mean, std, meta)

# One-hot encoding
X_train_norm_encoded = one_hot_encoding(X_train_norm, meta)
X_test_norm_encoded = one_hot_encoding(X_test_norm, meta)

# Include the bias unit of 1 at index 0
bias = np.ones((rows, 1), dtype=int)
X_train_final = np.hstack((bias, X_train_norm_encoded))

# Random weights for features and bias unit
w = np.random.uniform(low=-0.01, high=0.01, size=(1, cols+1))
