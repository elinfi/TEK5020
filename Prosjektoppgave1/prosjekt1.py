import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from min_err_rate import min_err_rate
from least_squares import least_squares
from nearest_neighbor import nearest_neighbor

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
    data = pd.read_csv(filename, header=None, sep='\s+', engine='python')
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    return data

def best_combination(err_rate, idx_array):
    best_idx = np.argmin(err_rate)
    best_comb = idx_array[best_idx]
    return err_rate[best_idx], best_comb

data = readfile('ds-3.txt')
train, test = train_test_split(data.to_numpy())

if train.shape[1] == 5:
    err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4, \
        idx_d1, idx_d2, idx_d3, idx_d4 = nearest_neighbor(test, train)

    error_rate = [err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4]
    index = [idx_d1, idx_d2, idx_d3, idx_d4]
else:
    err_rate_d1, err_rate_d2, err_rate_d3, \
        idx_d1, idx_d2, idx_d3 = nearest_neighbor(test, train)

    error_rate = [err_rate_d1, err_rate_d2, err_rate_d3]
    index = [idx_d1, idx_d2, idx_d3]

for err_rate, idx in zip(error_rate, index):
    NN_err_rate, best_comb = best_combination(err_rate, idx)
    minimum_err_rate = min_err_rate(test, train, best_comb)
    ls_err_rate = least_squares(test, train, best_comb)

    print('-------------------------------------------')
    print(f'Best combination: \t {best_comb}')
    print(f'Nearest neighbor: \t {NN_err_rate:.3}')
    print(f'Minimum error rate: \t {minimum_err_rate:.3}')
    print(f'Least squares: \t \t {ls_err_rate:.3}')
    print('-------------------------------------------\n')

print(err_rate_d1)
print(err_rate_d2)
print(err_rate_d3)
print(err_rate_d4)
