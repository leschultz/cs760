#!/usr/bin/env python3.6

import DecisionTree as dt

import numpy as np

import json
import sys

if __name__ == "__main__":
    np.random.seed(0)

argsmaxtrees = int(sys.argv[1])
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
w = np.ones(X_train.shape[0])/np.arange(1, X_train.shape[0]+1)

t = 0
while t < argsmaxtrees:

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

    # Compute the error
    e = np.sum(w*(y_pred != y_train))/np.sum(w)

    # Compute alpha
    alpha = np.log((1-e)/e)+np.log(nclasses-1)
    t += 1

indices = []
probs = []
preds = []
for tree in range(argsmaxtrees):

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
    preds.append(y_pred.astype(np.object))
    probs.append(y_prob.astype(np.object))

indices = np.column_stack(indices)


'''
combined = np.apply_along_axis(np.argmax, 1, np.sum(np.array(probs), axis=0))
predict = []
for i in combined:
    prediction = test_meta[-1][1][i]
    predict.append(prediction)

predict = np.array(predict)

accuracy = len(y_test[predict == y_test])/len(y_test)

for tree in indices:
    out = ','.join(map(str, tree))
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
      '{}'.format(np.round(accuracy, 16)),
      file=sys.stdout
      )
'''
