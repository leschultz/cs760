#!/usr/bin/env python3.6

from functions import *

import DecisionTree as dt

import numpy as np

import argparse

parserdescription = 'Bootstrap Aggregation (Bagging)'

parser = argparse.ArgumentParser(
                                 description=parserdescription
                                 )
parser.add_argument(
                    'trees',
                    type=float,
                    help='The learning rate'
                    )
parser.add_argument(
                    'max_depth',
                    type=int,
                    help='The number of hidden units'
                    )
parser.add_argument(
                    'train',
                    type=int,
                    help='Path to training set'
                    )
parser.add_argument(
                    'test',
                    type=str,
                    help='Path to testing set'
                    )

args = parser.parse_args()

# Load the training and testing data
X_train, y_train, X_test, y_test, meta = load(args.train, args.test)

# Train model
predictor = dt.DecisionTree()
predictor.fit(X_train, y_train, meta, max_depth=args.max_depth)

# Predict
predicted_y - predictor.predict(X_test, prob=True)
