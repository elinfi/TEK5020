import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nearest_neighbor import nearest_neighbor
from min_err_rate import min_err_rate

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
    data = pd.read_csv(filename, header=None, sep='\s+', engine='python')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(data)
    # print(data.to_numpy())
    return data





data = readfile('ds-3.txt')
train, test = train_test_split(data.to_numpy())

err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4 = nearest_neighbor(test, train)
print(err_rate_d1)
print(err_rate_d2)
print(err_rate_d3)
print(err_rate_d4)

min_err_rate(test, train, [1])
