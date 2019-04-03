#!/usr/bin/env python3.6

import DecisionTree as dt

import numpy as np

import json
import sys

argsmax_trees = int(sys.argv[1])
argsmax_depth = int(sys.argv[2])
argstrain = sys.argv[3]
argstest = sys.argv[4]

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
                  max_depth=argsmax_depth,
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
    probs.append(predictor.predict(X_test, prob=True))

weights = np.column_stack(weights)

combined = np.apply_along_axis(np.argmax, 1, np.sum(np.array(probs), axis=0))
predict = []
for i in combined:
    prediction = test_meta[-1][1][i]
    predict.append(prediction)

predict = np.array(predict)

accuracy = len(y_test[predict == y_test])/len(y_test)

for tree in weights:
    tree = ['{:.12f}'.format(np.round(i, 12)) for i in tree]
    out = ','.join(map(str, tree))
    print(out, file=sys.stdout)

print(file=sys.stdout)

treeweights =['{:.12f}'.format(np.round(i, 12)) for i in treeweights]
out = ','.join(map(str, treeweights))
print(out, file=sys.stdout)

print(file=sys.stdout)

preds.append(predict.astype(np.object))
preds.append(y_test.astype(np.object))
preds = np.column_stack(preds)
for tree in preds:
    out = ','.join(map(str, tree))
    print(out, file=sys.stdout)

print(file=sys.stdout)

print(
      '{:.12f}'.format(np.round(accuracy, 16)),
      file=sys.stdout
      )
