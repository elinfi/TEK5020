import numpy as np

def g(x, W, w, w0):
    """ Calcualates the quadratic discriminant function.

    Keyword arguments:
    x -- 1xd feature vector
    W -- dxd matrix
    w -- dx1 vector
    w0 -- 1x1 scalar

    Return value:
    the result of the quadratic discriminant function
    """
    return x @ W @ x.T + w.T @ x.T + w0

def min_err_rate(test, train1, train2, train3):
    """
    Calculate the error rate for the best feature combination using the minimum
    error rate.

    Keyword arguments:
    test -- input test data
    train1 -- input train data for class 1
    train2 -- input train data for class 2
    train3 -- input train data for class 3

    Return value:
    segmentation -- segmented test data based on train1, train2, train3

    Other:
    Every variable name ended with 1, 2 or 3 refers to class 1, 2 or 3
    respectively.
    """


    # number of objects
    n = test.shape[0] + test.shape[1]
    n1 = train1.shape[0] + train1.shape[1]
    n2 = train2.shape[0] + train2.shape[1]
    n3 = train3.shape[0] + train3.shape[1]

    # a priori probability for each class
    P_omega1 = n1/n
    P_omega2 = n2/n
    P_omega3 = n3/n

    # maximum likelihood estimate of the expectation value for each class
    mu1 = 1/n1 * np.sum(np.sum(train1, axis=0), axis=0)
    mu2 = 1/n2 * np.sum(np.sum(train2, axis=0), axis=0)
    mu3 = 1/n3 * np.sum(np.sum(train3, axis=0), axis=0)

    diff1 = (train1 - mu1)
    diff2 = (train2 - mu2)
    diff3 = (train3 - mu3)
    cov1 = np.zeros((3, 3))
    cov2 = np.zeros((3, 3))
    cov3 = np.zeros((3, 3))

    for i in range(train1.shape[0]):
        for j in range(train1.shape[1]):
            cov1 += (diff1[i, j].reshape(-1, 1) @ diff1[i, j].reshape(-1, 1).T)
    for i in range(train2.shape[0]):
        for j in range(train2.shape[1]):
            cov2 += (diff2[i, j].reshape(-1, 1) @ diff2[i, j].reshape(-1, 1).T)
    for i in range(train3.shape[0]):
        for j in range(train3.shape[1]):
            cov3 += (diff3[i, j].reshape(-1, 1) @ diff3[i, j].reshape(-1, 1).T)
    cov1 /= n1
    cov2 /= n2
    cov3 /= n3

    # print(np.linalg.det(cov1))
    # print(np.linalg.det(cov2))
    # print(np.linalg.det(cov3))

    W1 = -1/2 * np.linalg.pinv(cov1)
    W2 = -1/2 * np.linalg.pinv(cov2)
    W3 = -1/2 * np.linalg.pinv(cov3)

    w1 = np.linalg.pinv(cov1) @ mu1.T
    w2 = np.linalg.pinv(cov2) @ mu2.T
    w3 = np.linalg.pinv(cov3) @ mu3.T

    w10 = -1/2 * mu1 @ np.linalg.pinv(cov1) @ mu1.T \
          - 1/2 * np.log(np.linalg.det(cov1)) + np.log(P_omega1)
    w20 = -1/2 * mu2 @ np.linalg.pinv(cov2) @ mu2.T \
          - 1/2 * np.log(np.linalg.det(cov2)) + np.log(P_omega2)
    w30 = -1/2 * mu3 @ np.linalg.pinv(cov3) @ mu3.T \
          - 1/2 * np.log(np.linalg.det(cov3)) + np.log(P_omega3)

    segmentation = np.zeros(test.shape)
    RGB = [[1, 0.8, 0.3], [1, 0.7, 0.9], [0.1, 0.8, 0.7]]
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            g1 = g(test[i, j], W1, w1, w10)
            g2 = g(test[i, j], W2, w2, w20)
            g3 = g(test[i, j], W3, w3, w30)
            g_arr = np.array([g1, g2, g3])
            idx = np.argmax(np.array([g1, g2, g3]))
            segmentation[i, j] = RGB[idx]

    return segmentation
