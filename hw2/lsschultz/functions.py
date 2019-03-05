'''
This script standardizes the data for machine learning.
'''

import numpy as np


def datasplitter(x):
    '''
    Split data and classes for a dataset.

    inputs:
        x = The data containing features with the last column being the class

    outputs:
        data = The feature data
        class = The class label
    '''

    array = np.array(x)
    data = array[:, :-1]

    classes = []
    for i in x:
        classes.append(i[-1])

    return data, classes


def Bayes_Rule(x, true_class):
    '''
    Calculate probabilities based on Bayes' Rule.

    inputs:
        x = Both the data and metadata

    outputs:
    '''
