#!/usr/bin/env python3.6

import DecisionTree as dt

import numpy as np

import json
import sys

if __name__ == "__main__":
    np.random.seed(0)

argsntrees = int(sys.argv[1])
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

indices = []
probs = []
preds = []
for tree in range(argsntrees):

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
                  max_depth=argsmax_depth
                  )

    # Predict
    y_prob = predictor.predict(X_test, prob=True)
    y_pred = predictor.predict(X_test)

    indices.append(choice)
    preds.append(y_pred)
    probs.append(y_prob)

indices = np.column_stack(indices)

combined = np.apply_along_axis(np.argmax, 1, np.sum(np.array(probs), axis=0))
predict = []
for i in combined:
    prediction = test_meta[-1][1][i]
    predict.append(prediction)

accuracy = len(y_test[predict == y_test])/len(y_test)

for tree in indices:
    out = ','.join(map(str, tree))
    print(out, file=sys.stdout)

print(file=sys.stdout)

preds.append(predict)
preds.append(y_test)
preds = np.column_stack(preds)
for tree in preds:
    out = ','.join(map(str, tree))
    print(out, file=sys.stdout)

print(file=sys.stdout)

print(
      '{}'.format(np.round(accuracy, 16)),
      file=sys.stdout
      )
