#!/usr/bin/env python3.6

from functions import *

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

parser.add_argument(
                    'nt',
                    type=str,
                    help='n for Naive Bayes or t for TAN'
                    )

args = parser.parse_args()


# The option for Naive Bayes or TAN
nt = args.nt

# Load the training and testing data
X_train, y_train, X_test, y_test, meta = load(args.train, args.test)

if nt == 'n':
    probabilities = naive_bayes(X_train, y_train, X_test, y_test, meta)
    print_info(probabilities)

if nt == 't':
    probabilities = tan(X_train, y_train, X_test, y_test, meta)
    #print_info(probabilities)
