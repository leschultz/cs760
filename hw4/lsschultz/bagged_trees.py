#!/usr/bin/env python3.6

import DecisionTree as dt

import numpy as np

import argparse
import json

parserdescription = 'Bootstrap Aggregation (Bagging)'

parser = argparse.ArgumentParser(
                                 description=parserdescription
                                 )
parser.add_argument(
                    'ntrees',
                    type=int,
                    help='The number of trees'
                    )
parser.add_argument(
                    'max_depth',
                    type=int,
                    help='The maximum depth of trees'
                    )
parser.add_argument(
                    'train',
                    type=str,
                    help='Path to training set'
                    )
parser.add_argument(
                    'test',
                    type=str,
                    help='Path to testing set'
                    )

args = parser.parse_args()

# Load the training and testing data
train = json.load(open(args.train, 'r'))
test = json.load(open(args.train, 'r'))

train_meta = train['metadata']['features']
train_data = np.array(train['data'])

test_meta = test['metadata']['features']
test_data = np.array(test['data'])

X_train = train_data[:, :-1]
y_train = train_data[:, -1]

X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Train model
predictor = dt.DecisionTree()
predictor.fit(X_train, y_train, train_meta, max_depth=args.max_depth)

# Predict
y_pred = predictor.predict(X_test, prob=True)
