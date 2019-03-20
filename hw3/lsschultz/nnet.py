#!/usr/bin/env python3.6

from functions import *

import numpy as np

import argparse

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
                    'hidden',
                    type=int,
                    help='The number of hidden units'
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

# Normalize the numeric features
X_train_norm, mean, std = trainnorm(X_train, meta)
X_test_norm = testnorm(X_test, mean, std, meta)

# One-hot encoding
X_train_norm_encoded = one_hot_encoding(X_train_norm, meta)
X_test_norm_encoded = one_hot_encoding(X_test_norm, meta)

# The number of rows and columns for training data excluding the classes
X_train_rows, X_train_cols = X_train_norm_encoded.shape
X_test_rows, X_test_cols = X_test_norm_encoded.shape

# Include the bias unit of 1 at index 0
X_train_final = np.hstack((
                           np.ones((X_train_rows, 1), dtype=int),
                           X_train_norm_encoded
                           ))
X_test_final = np.hstack((
                          np.ones((X_test_rows, 1), dtype=int),
                          X_test_norm_encoded
                          ))

# Turn target data into binary form
y_train_binary = binarytarget(y_train, meta)
y_test_binary = binarytarget(y_test, meta)

# Random weights for the bias unit and features
np.random.seed(0)
w_i_h = np.random.uniform(low=-0.01, high=0.01, size=(args.hidden, X_train_cols+1))
w_h_o = np.random.uniform(low=-0.01, high=0.01, size=(1, args.hidden+1))

# Apply logistic regression for training
epochs = [i+1 for i in list(range(args.epochs))]
nhidden = [i+1 for i in list(range(args.hidden))]

w_i_h, w_h_o, errors, ncorrect, nincorrect = nnepoch(
                                                     X_train_final,
                                                     y_train_binary,
                                                     nhidden,
                                                     w_i_h,
                                                     w_h_o,
                                                     args.rate,
                                                     epochs
                                                     )
threshold = 0.5
result, activations, testcorrect, testincorrect = nnpredict(
                                                            X_test_final,
                                                            y_test_binary,
                                                            w_i_h,
                                                            w_h_o,
                                                            threshold
                                                            )
print(result)
f1 = f1_score(result, y_test_binary)

lr_print(
         epochs,
         errors,
         ncorrect,
         nincorrect,
         activations,
         result,
         y_test_binary,
         testcorrect,
         testincorrect,
         f1
         )
