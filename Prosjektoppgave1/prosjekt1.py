import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_test_split(data):
    # create boolean index array with True for every even index and False for
    # every odd index
    index = np.linspace(1, len(data), len(data))
    boolean = index % 2 == 0

    # split data in train and test
    train = data[np.invert(boolean)]
    test = data[boolean]

    return train, test

def readfile(filename):
    # data = pd.read_csv(filename, header=None, sep='  ', index_col=0)
    data = pd.read_csv(filename, header=None, sep='  ', engine='python')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(data)
    # print(data.to_numpy())
    return data

def nearest_neighbor(data):
    max_dim = data.shape[1] - 1

    for d in range(1, max_dim + 1):
        train_object = train[:, 1:]
        test_object = test[:, 1:]

        if d == 1:
            # Find the distance from test object to train object for all features
            distance = np.abs(test_object[:, np.newaxis] - train_object)

            # Find the index of the nearest neighbor for each feature
            index = np.argmin(distance, axis=1)

            # Get the class of the nearest neighbor for each feature
            NN_class = train[index, 0]

            # Find failed classified elements for each feature
            boolean = NN_class != test[:, 0].reshape(test.shape[0], 1)

            # Calculate error rate for each feature
            err_rate = np.mean(boolean, axis=0)

        elif d == max_dim:
            # Find the distance from test object to train object using all features
            distance = np.linalg.norm(test_object[:, np.newaxis] - train_object, axis=1, keepdims=True)

            # Find the index of the nearest neighbor
            index = np.argmin(distance, axis=1)

            # Get the class of the nearest neighbor
            NN_class = train[index, 0]

            # Find failed classified elements
            boolean = NN_class != test[:, 0].reshape(test.shape[0], 1)

            # Calculate error rate for each feature
            err_rate = np.mean(boolean, axis=0)

        elif d == 2:
            for i in range(1, maxdim + 1):
                for j in range(i, maxdim + 1):
                    distance = np.linalg.norm()



data = readfile('ds-1.txt')
train, test = train_test_split(data.to_numpy())


train_object = train[:, 1:]
test_object = test[:, 1:]


# train_objects = train[:, 1:]
# NN_class = np.empty(test.shape[0])
# for i in range(test.shape[0]):
#     test_object = test[i, 1:]
#     distance = np.linalg.norm(train_objects - test_object, axis=1, keepdims=True)
#     index = np.argmin(distance)
#     NN_class[i] = int(train[index, 0])
#
# test_class = test[:, 0]
#
# n_wrong = np.sum(test_class - NN_class != 0)
# n_total = len(test)
# error_rate = n_wrong/n_total
# print(error_rate)
