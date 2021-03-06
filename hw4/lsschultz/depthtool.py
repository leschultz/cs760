from matplotlib import pyplot as pl

import DecisionTree as dt

import numpy as np

import json
import sys

name_bag = 'bagged_tree_plot.pdf'
name_boost = 'boosted_tree_plot.pdf'

method = 'boost'
argsmax_trees = 5
argsmax_depth = list(range(1, 11))
argstrain = '../datasets/digits_train.json'
argstest = '../datasets/digits_test.json'

# Load the training and testing data
train = json.load(open(argstrain, 'r'))
test = json.load(open(argstest, 'r'))

train_meta = train['metadata']['features']
train_data = np.array(train['data'])

test_meta = test['metadata']['features']
test_data = np.array(test['data'])

X_train = train_data[:, :-1]
y_train = train_data[:, -1]

X_test = test_data[:, :-1]
y_test = test_data[:, -1]

nclasses = len(train_meta[-1][-1])

acc = []
for depth in argsmax_depth:

    # Run bagged-trees
    if method == 'bag':

        np.random.seed(0)

        indices = []
        probs = []
        preds = []
        for tree in range(argsmax_trees):

            choice = np.random.choice(
                                      X_train.shape[0],
                                      X_train.shape[0],
                                      )

            X_train_strap = X_train[choice, :]
            y_train_strap = y_train[choice]

            # Train model
            predictor = dt.DecisionTree()
            predictor.fit(
                          X_train_strap,
                          y_train_strap,
                          train_meta,
                          max_depth=depth
                          )

            # Predict
            y_prob = predictor.predict(X_test, prob=True)
            y_pred = predictor.predict(X_test)

            indices.append(choice)
            preds.append(y_pred.astype(np.object))
            probs.append(y_prob.astype(np.object))

        indices = np.column_stack(indices)

        combined = np.apply_along_axis(np.argmax, 1, np.sum(np.array(probs), axis=0))
        predict = []
        for i in combined:
            prediction = test_meta[-1][1][i]
            predict.append(prediction)

        predict = np.array(predict)

    # Run boosted-trees
    if method == 'boost':

        # Initialize instance weights
        w = np.ones(X_train.shape[0])/X_train.shape[0]

        weights = []
        treeweights = []
        preds = []
        probs = []
        for tree in range(argsmax_trees):

            # Normalize weights
            w = w/sum(w)

            weights.append(w)

            # Train model
            predictor = dt.DecisionTree()
            predictor.fit(
                          X_train,
                          y_train,
                          train_meta,
                          max_depth=depth,
                          instance_weights=w
                          )

            # Predict on training set
            y_pred = predictor.predict(X_train)

            # Find matching values
            I = (y_pred != y_train)

            # Compute the error
            e = sum(w*I)/sum(w)

            if e >= 1-1/nclasses:
                break

            # Compute alpha
            alpha = np.log((1-e)/e)+np.log(nclasses-1)

            # Update weights
            w = w*np.exp(alpha*I)

            preds.append(predictor.predict(X_test))
            treeweights.append(alpha)
            probs.append(predictor.predict(X_test, prob=True)*alpha)

        weights = np.column_stack(weights)
        probs = np.array(probs)

        predict = []
        for row in np.array(preds).T:

            count = 0
            votes = {}
            for item in train_meta[-1][-1]:
                votes[item] = sum([(i==item)*j for i, j in zip(row, treeweights)])

                count += 1

            voteindex = max(votes, key=votes.get)
            predict.append(voteindex)

        predict = np.array(predict)

    accuracy = len(y_test[predict == y_test])/len(y_test)
    acc.append(accuracy)
    print(accuracy)
