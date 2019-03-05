#!/usr/bin/env python3.6

from functions import *

import numpy as np

import argparse
import json
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

# Open the training and test data
train = open(args.train)
test = open(args.test)

# Load the data in opened files
train = json.load(train)
test = json.load(test)

# The target feature is the lasat feature named 'label'
train_data, train_classes = datasplitter(train['data'])
test_data, test_classes = datasplitter(test['data'])

# The data types
train_types = [i[-1] for i in train['metadata']['features'][:-1]]
test_types = [i[-1] for i in test['metadata']['features'][:-1]]

# The available class labels
labels = train['metadata']['features'][-1][-1]

print(labels)


'''
# Print the data in standard output
for item in results:
    out = ','.join(map(str, item))
    print(out, file=sys.stdout)
'''
