#!/usr/bin/env python3.6

from functions import *

import argparse
import sys

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
dftrain, dftest, labels = load(args.train, args.test)


if nt == 'n':
    naive_bayes(dftrain, dftest, labels)

'''
# Print the data in standard output
for item in results:
    out = ','.join(map(str, item))
    print(out, file=sys.stdout)
'''
