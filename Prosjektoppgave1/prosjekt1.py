import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nearest_neighbor import nearest_neighbor2

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





data = readfile('ds-1.txt')
train, test = train_test_split(data.to_numpy())

nearest_neighbor2(test, train)


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
