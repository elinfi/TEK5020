import math
import numpy as np
from scipy.special import comb

def nearest_neighbor(train, test, dimension):
    max_dim = train.shape[1] - 1
    print(max_dim)
    train_object = train[:, 1:]
    test_object = test[:, 1:]

    if dimension == 1:
        # Find the distance from test object to train object
        distance = np.abs(test_object - train_object)
        print(distance.shape)

        # Find the index of the nearest neighbor
        index = np.argmin(distance, axis=1)
        print(index)

        # Get the class of the nearest neighbor
        NN_class = train[index, 0]

        # Find failed classified elements
        boolean = NN_class != test[:, 0].reshape(-1, 1)

        # Calculate error rate for each feature
        err_rate = np.mean(boolean, axis=0)
        print(err_rate)

        # Find the combination with the least error
        best_combination = np.argmin(err_rate)
        print(best_combination)

    elif dimension == 2:
        n_combinations = int(math.factorial(max_dim)/(math.factorial(dimension)*math.factorial(max_dim - dimension)))
        distance = np.zeros((train.shape[0], n_combinations))
        idx = 0
        combinations = np.zeros((n_combinations, 2))
        for i in range(max_dim):
            for j in range(i + 1, max_dim):
                combinations[idx] = [i, j]
                # Find the distance from test object to train object using all features
                distance[:, idx] = np.linalg.norm(test_object[:, [i, j]] - train_object[:, [i, j]], axis=1)
                idx += 1
        # Find the index of the nearest neighbor
        index = np.argmin(distance, axis=1)

        # Get the class of the nearest neighbor
        NN_class = train[index, 0]

        # Find failed classified elements
        boolean = NN_class != test[:, 0].reshape(test.shape[0], 1)

        # Calculate error rate for each feature
        err_rate = np.mean(boolean, axis=0)

        # Find the combination with the least error
        best_combination = combinations[np.argmin(err_rate)]

        print(err_rate)
        print(best_combination)

    elif dimension == 3:
        distance = np.zeros((train.shape[0], 4))
        combinations = np.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])

        distance[:, 0] = np.linalg.norm(test_object[:, [0, 1, 2]] - train_object[:, [0, 1, 2]], axis=1)
        distance[:, 1] = np.linalg.norm(test_object[:, [0, 1, 3]] - train_object[:, [0, 1, 3]], axis=1)
        distance[:, 2] = np.linalg.norm(test_object[:, [0, 2, 3]] - train_object[:, [0, 2, 3]], axis=1)
        distance[:, 3] = np.linalg.norm(test_object[:, [1, 2, 3]] - train_object[:, [1, 2, 3]], axis=1)

        # Find the index of the nearest neighbor
        index = np.argmin(distance, axis=1)
        print(index)

        # Get the class of the nearest neighbor
        NN_class = train[index, 0]

        # Find failed classified elements
        boolean = NN_class != test[:, 0].reshape(test.shape[0], 1)

        # Calculate error rate for each feature
        err_rate = np.mean(boolean, axis=0)

        # Find the combination with the least error rate
        best_combination = combinations[np.argmin(err_rate)]

        print(err_rate)
        print(best_combination)

    elif dimension == max_dim:
        # Find the distance from test object to train object using all features
        distance = np.linalg.norm(test_object - train_object, axis=1, keepdims=True)

        # Find the index of the nearest neighbor
        index = np.argmin(distance, axis=1)

        # Get the class of the nearest neighbor
        NN_class = train[index, 0]

        # Find failed classified elements
        boolean = NN_class != test[:, 0].reshape(test.shape[0], 1)

        # Calculate error rate for each feature
        err_rate = np.mean(boolean, axis=0)

        print(err_rate)
        print('hei')

    # return err_rate



def nearest_neighbor1(data):
    max_dim = data.shape[1] - 1

    err_rate

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

        elif d == 2:
            distance = zeros(train.shape[0], math.factorial(max_dim - 1))
            idx = 0
            for i in range(1, maxdim + 1):
                for j in range(i + 1, maxdim + 1):
                    idx += 1
                    # Find the distance from test object to train object using all features
                    distance[:, idx] = np.linalg.norm(test_object[:, [i, j]] - train_object[:, [i, j]], axis=1, keepdims=True)

            # Find the index of the nearest neighbor
            index = np.argmin(distance, axis=1)

            # Get the class of the nearest neighbor
            NN_class = train[index, 0]

            # Find failed classified elements
            boolean = NN_class != test[:, 0].reshape(test.shape[0], 1)

            # Calculate error rate for each feature
            err_rate = np.mean(boolean, axis=0)

        elif d == 3 & maxdim == 4:
            distance = zeros(train.shape[0], 4)

            distance[:, 0] = np.linalg.norm(test_object[:, [1, 2, 3]] - train_object[:, [i, j]], axis=1, keepdims=True)
            distance[:, 1] = np.linalg.norm(test_object[:, [1, 2, 4]] - train_object[:, [i, j]], axis=1, keepdims=True)
            distance[:, 2] = np.linalg.norm(test_object[:, [1, 3, 4]] - train_object[:, [i, j]], axis=1, keepdims=True)
            distance[:, 3] = np.linalg.norm(test_object[:, [2, 3, 4]] - train_object[:, [i, j]], axis=1, keepdims=True)

            # Find the index of the nearest neighbor
            index = np.argmin(distance, axis=1)

            # Get the class of the nearest neighbor
            NN_class = train[index, 0]

            # Find failed classified elements
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


def nearest_neighbor2(test_data, train_data):
    test_objects = test_data[:, 1:]
    train_objects = train_data[:, 1:]

    features = test_objects.shape[1]
    err_rate_d1 = np.zeros(features)
    err_rate_d2 = np.zeros(comb(features, 2, exact=True))
    err_rate_d3 = np.zeros(comb(features, 3, exact=True))
    err_rate_d4 = np.zeros(1)

    test_idx = 0
    for test in test_objects:
        d2_idx = 0
        d3_idx = 0
        for i in range(features):
            difference = np.abs(test[i] - train_objects[:, i])
            index = np.argmin(difference)
            class_error = test_data[test_idx, 0] != train_data[index, 0]
            err_rate_d1[i] += class_error

            for j in range(i + 1, features):
                difference = np.linalg.norm(test[[i, j]] - train_objects[:, [i, j]], axis=1)
                index = np.argmin(difference)
                class_error = test_data[test_idx, 0] != train_data[index, 0]
                err_rate_d2[d2_idx] += class_error
                d2_idx += 1

                for k in range(j + 1, features):
                    difference = np.linalg.norm(test[[i, j, k]] - train_objects[:, [i, j, k]], axis=1)
                    index = np.argmin(difference)
                    class_error = test_data[test_idx, 0] != train_data[index, 0]
                    err_rate_d3[d3_idx] += class_error
                    d3_idx += 1

                    for l in range(k + 1, features):
                        difference = np.linalg.norm(test[[i, j, k, l]] - train_objects[:, [i, j, k, l]], axis=1)
                        index = np.argmin(difference)
                        class_error = test_data[test_idx, 0] != train_data[index, 0]
                        err_rate_d4[0] += class_error
        test_idx += 1
    n_objects = test_objects.shape[0]
    err_rate_d1 /= n_objects
    err_rate_d2 /= n_objects
    err_rate_d3 /= n_objects
    err_rate_d4 /= n_objects

    print(err_rate_d1)
    print(err_rate_d2)
    print(err_rate_d3)
    print(err_rate_d4)
