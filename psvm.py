# Written by Calen Irwin & Ryland Willhans
# Written for COIS-4350H
# Last Modified Date: 2020-04-14
#
# Purpose: This program uses the MPI standard to parallelize an SVM classifier. The cost reduction step of our SVM is the portion 
#          that is parallelized. Each process is given a relatively equal subset of the data, and then it uses SGD to calculate 
#          the optimal weights (which define the classifier hyperplane) for a given subset of data. The root process then aggregates 
#          the calculated weights from all processes to achive the final weights which will be used to determine our model's predicted 
#          class labels.
#          
# Instructions for running: *depending on your version of Python use 'python' instead of 'py'*
#   To run with MPI    >> 'mpiexec -n {number of processes} py -m mpi4py psvm.py {inputfile.csv} {class_label} {+ve class value} {-ve class value}'
# References: [0] https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2

import sys  # library for accessing command line arguments
import math # library for basic mathematical operations
import numpy as np # package for scientific computing
import pandas as pd # package for data analysis and manipulation
import matplotlib.pyplot as plt # package for displaying confusion matrix
import scikitplot as skplt  # package for generating confusion matrix
import statsmodels.api as sm    # package for statistical modelling
from sklearn.preprocessing import MinMaxScaler  # used to normalize data
from sklearn.model_selection import train_test_split as tts # used to divide training and test data
from sklearn.metrics import accuracy_score, recall_score, precision_score, auc  # used to get statistical information on the model's performance
from sklearn.utils import shuffle   # used to shuffle data before training
from time import process_time   # used to time training and testing model
from mpi4py import MPI  # package for MPI 


def init():
    if len(sys.argv) == 5:  # expects 4 additional command-line arguments
        data_file = sys.argv[1]
        class_label = sys.argv[2]
        # check to see if the class values are numerical
        # and casts them to ints if needed
        if (is_intstring(sys.argv[3])):
            positive_case = int(sys.argv[3])
        else:
            positive_case = sys.argv[3]

        if (is_intstring(sys.argv[4])):
            negative_case = int(sys.argv[4])
        else:
            negative_case = sys.argv[4]

        comm = MPI.COMM_WORLD   # MPI process group variable
        size = comm.Get_size()  # size = total number of processes 
        rank = comm.Get_rank()  # rank = ID of current process (e.g., 0, 1, 2, etc.)

        if rank == 0:   # if the process is the root process
            data = pd.read_csv('./' + data_file)    # read in data file
            start = process_time()                  # start timer

            # transform class values to -1 for -ve case and 1 for +ve case
            # note: SVMs only take in numerical data
            data[class_label] = data[class_label].map({negative_case:-1.0, positive_case:1.0})
            
            # partition label and features 
            Y = data.loc[:, class_label]
            cols = data.columns.tolist()
            cols.remove(class_label)
            X = data.loc[:, cols]

            # preprocessing of data to yield better results
            remove_correlated_features(X)  
            remove_less_significant_features(X, Y) 

            # min-max normalization of data values
            X_normalized = MinMaxScaler().fit_transform(X.values)

            X = pd.DataFrame(X_normalized)  # convert to pandas dataframe

            X.insert(loc=len(X.columns), column='intercept', value=1)
        
            print("Splitting dataset into training and test sets...")

            # test_size = portion of data to reserve for the test set
            # random_state = seed for random number generator
            X_train, X_test, Y_train, Y_test = tts(X, Y, test_size = 0.2, random_state = 42)

            X_train = X_train.to_numpy()    # convert to numpy array
            Y_train = Y_train.to_numpy()    # convert to numpy array

            num_cols = X_train.shape[1]     # get the number of columns

            split_size = len(X_train) / size            # number of rows per proces if there was no remainder

            num_rows = [math.floor(split_size)] * size  # number of rows each process will receive

            for i in range(len(X_train) % size):        # add the remaining rows among the processes
                num_rows[i] += 1

            X_train = np.ascontiguousarray(X_train)     # convert the array back into row-major format

            final_weights = np.zeros(num_cols)          # array to hold weights after they've been gathered from all processes        

            split_sizes = [num_cols * x for x in num_rows]  # for Scatterv to know the number of rows to give each process
        
            Y_displacements = [0] * size    # make an array of 0s to hold the displacements for Y values

            # calculate Y displacements
            for i in range(1, size):
                Y_displacements[i] = Y_displacements[i - 1] + num_rows[i - 1]

            # use Y displacements to calculate X displacements
            X_displacements = [x * num_cols for x in Y_displacements]

        # instantiate variables on non-root processes
        else:
            num_rows = None
            num_cols = None
            split_sizes = None
            X_train = None
            Y_train = None
            final_weights = None
            X_displacements = None
            Y_displacements = None

        # broadcast row/column information to all processes
        num_rows = comm.bcast(num_rows, root=0)
        num_cols = comm.bcast(num_cols, root=0)

        X_buf = np.empty((num_rows[rank], num_cols))    # buffer the X values that each process will receive
        Y_buf = np.empty(num_rows[rank])                # buffer the Y values that each process will receive

        # use Scatterv opposed to Scatter to send a 2D row-major array as a contiguous block of memory
        # split_sizes is a list of counts
        # X and Y displacements
        comm.Scatterv([X_train, split_sizes, X_displacements, MPI.DOUBLE], X_buf, root=0)
        comm.Scatterv([Y_train, num_rows, Y_displacements, MPI.DOUBLE], Y_buf, root=0)
        
        print(f"Process {rank} has begun training...")
        # build the model
        W = sgd(X_buf, Y_buf)
        print(f"Process {rank} has finished training.")
        print(f"Weights: {W}")

        # gather result vectors
        comm.Reduce([W, MPI.DOUBLE], [final_weights, MPI.DOUBLE], op = MPI.SUM, root=0)

        if rank == 0:
            final_weights = [x/size for x in final_weights]          # average weights 
            print(f"Final weights: {final_weights}")

            stop = process_time()                                    # stop timer
            print(f"Time taken for training: {stop - start}")

            Y_test_predicted = np.sign(np.dot(X_test.to_numpy(), final_weights)) # apply model to get predicted classes
            Y_test = Y_test.to_numpy()                               # actual classes

            # calculate and output results
            accuracy = accuracy_score(Y_test, Y_test_predicted)
            recall = recall_score(Y_test, Y_test_predicted)
            precision = precision_score(Y_test, Y_test_predicted)

            print(f"Accuracy on test dataset: {accuracy}")
            print(f"Recall on test dataset: {recall}")
            print(f"Precision on test dataset: {precision}")

            skplt.metrics.plot_confusion_matrix(Y_test, Y_test_predicted, normalize=True)
            plt.show()
    else: 
        print("***Incorrect arguments, proper format: py ./psvm.py {data filename} {class label} {positive class value} {negative class value}")

# taken from reference [0]
def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)
    
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

# taken from reference [0]
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

# stochastic gradient descent function
# purpose: to minimize the cost function of the SVM
# taken from reference [0]
def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch {}, Cost = {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights

# preprocessing of data to yield better results
# taken from reference [0]
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped

# preprocessing of data to yield better results
# taken from reference [0]
def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

# convert to int if possible
def is_intstring(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

reg_strength = 10000            # regularization strength
learning_rate = 0.000001        # learning rate
init()                          # initialize the program