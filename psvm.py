# Written by Calen Irwin & Ryland Willhans
# Written for COIS-4350H
# Last Modified Date: 
# Purpose: 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score 
from sklearn.utils import shuffle
from time import process_time


def init():
    start = process_time()
    data = pd.read_csv('./creditcard.csv')
    # SVM only accepts numerical values. 
    # Therefore, we will transform the categories M and B into
    # values 1 and -1 (or -1 and 1), respectively.
    data['Class'] = data['Class'].map({0:-1.0, 1:1.0})
    # print(data)
    Y = data.loc[:, 'Class']
    X = data.iloc[:, :-1]
    # normalize the features using MinMaxScalar from
    # sklearn.preprocessing
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)
    # test_size is the portion of data that will go into test set
    # random_state is the seed used by the random number generator
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

    # train the model
    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))
    y_test_predicted = np.array([])

    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i])) #model
        y_test_predicted = np.append(y_test_predicted, yp)
    stop = process_time()
    print("time taken: {}".format(stop-start))
    print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
    print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))

def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)
    
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)  # average
    return dw

def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights

# def remove_correlated_features(X):
#     corr_threshold = 0.9
#     corr = X.corr()
#     drop_columns = np.full(corr.shape[0], False, dtype=bool)
#     for i in range(corr.shape[0]):
#         for j in range(i + 1, corr.shape[0]):
#             if corr.iloc[i, j] >= corr_threshold:
#                 drop_columns[j] = True
#     columns_dropped = X.columns[drop_columns]
#     X.drop(columns_dropped, axis=1, inplace=True)
#     return columns_dropped
# def remove_less_significant_features(X, Y):
#     sl = 0.05
#     regression_ols = None
#     columns_dropped = np.array([])
#     for itr in range(0, len(X.columns)):
#         regression_ols = sm.OLS(Y, X).fit()
#         max_col = regression_ols.pvalues.idxmax()
#         max_val = regression_ols.pvalues.max()
#         if max_val > sl:
#             X.drop(max_col, axis='columns', inplace=True)
#             columns_dropped = np.append(columns_dropped, [max_col])
#         else:
#             break
#     regression_ols.summary()
#     return columns_dropped

reg_strength = 10000 # regularization strength
learning_rate = 0.000001
init()