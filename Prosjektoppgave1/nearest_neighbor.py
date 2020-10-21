import numpy as np
from scipy.special import comb

def nearest_neighbor(test_data, train_data):
    """
    Calculate the error rate for all combinations of features using the nearest
    neighbor classification.
    """
    # remove the true class for easier computations
    test_objects = test_data[:, 1:]
    train_objects = train_data[:, 1:]


    features = test_objects.shape[1]    # number of features
    # arrays containg error rate for all combinations of features
    err_rate_d1 = np.zeros(features)
    err_rate_d2 = np.zeros(comb(features, 2, exact=True))
    err_rate_d3 = np.zeros(comb(features, 3, exact=True))
    if features == 4:
        err_rate_d4 = np.zeros(1)

    test_idx = 0
    for test in test_objects:
        d2_idx = 0
        d3_idx = 0
        for i in range(features):
            # error rate in one dimension
            difference = np.abs(test[i] - train_objects[:, i])
            index = np.argmin(difference)
            class_error = test_data[test_idx, 0] != train_data[index, 0]
            err_rate_d1[i] += class_error

            for j in range(i + 1, features):
                # error rate in two dimensions
                difference = np.linalg.norm(test[[i, j]] - train_objects[:, [i, j]], axis=1)
                index = np.argmin(difference)
                class_error = test_data[test_idx, 0] != train_data[index, 0]
                err_rate_d2[d2_idx] += class_error
                d2_idx += 1

                for k in range(j + 1, features):
                    # error rate in three dimensions
                    difference = np.linalg.norm(test[[i, j, k]] - train_objects[:, [i, j, k]], axis=1)
                    index = np.argmin(difference)
                    class_error = test_data[test_idx, 0] != train_data[index, 0]
                    err_rate_d3[d3_idx] += class_error
                    d3_idx += 1

                    for l in range(k + 1, features):
                        # error rate in four dimensions
                        difference = np.linalg.norm(test[[i, j, k, l]] - train_objects[:, [i, j, k, l]], axis=1)
                        index = np.argmin(difference)
                        class_error = test_data[test_idx, 0] != train_data[index, 0]
                        err_rate_d4[0] += class_error
        test_idx += 1

    n_objects = test_objects.shape[0]
    err_rate_d1 /= n_objects
    err_rate_d2 /= n_objects
    err_rate_d3 /= n_objects
    if features == 4:
        err_rate_d4 /= n_objects
        return err_rate_d1, err_rate_d2, err_rate_d3, err_rate_d4
    return err_rate_d1, err_rate_d2, err_rate_d3

    print(err_rate_d1)
    print(err_rate_d2)
    print(err_rate_d3)
    print(err_rate_d4)
