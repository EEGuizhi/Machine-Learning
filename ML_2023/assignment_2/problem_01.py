# Ridge & Lasso Regression
import numpy as np
from numpy.linalg import inv

LAMBDA_LIST = (0, 0.01, 0.1, 1, 10)
DATA_X = np.array([  # N = dim 0, M = dim 1
    (1, 1),
    (1, 3),
    (5, 2),
    (7, 4),
    (9, 2)
])
DATA_Y = np.array([  # N = dim 0
    2,
    5,
    3,
    4,
    8
])


def ridge_regression(x_train:np.ndarray, y_train:np.ndarray, lambda_param:float):
    """get optimal weights obtained by Ridge Regression.

    Args:
        x_train : ndim array of `[Phi_1, Phi_2, ..., Phi_N]`
        y_train : 1dim array of `[y_1, y_2, ..., y_N]`
        lambda_param : control variable to reduce search region of w
    """
    mean_x = np.mean(x_train, axis=0)
    mean_y = np.mean(y_train, axis=0)
    X_o = x_train - mean_x
    y_o = y_train - mean_y

    matW = X_o.T @ X_o
    for i in range(matW.shape[0]): matW[i, i] += lambda_param
    matW = inv(matW) @ X_o.T @ y_o
    w0 = mean_y - mean_x @ matW

    matW = np.insert(matW, 0, w0, axis=0)
    return matW


def ridge_regression_iterative(x_train:np.ndarray, y_train:np.ndarray, lambda_param:float, threshold:float=0.00001, max_iteration:int=1000):
    """same as the ridge_regression() function, but this is iterative.

    Args:
        x_train : ndim array of `[Phi_1, Phi_2, ..., Phi_N]`
        y_train : 1dim array of `[y_1, y_2, ..., y_N]`
        lambda_param : control variable to reduce search region of w
    """
    converge = False

    # Step 0. Give initial values (zeros)
    w0 = np.array([0])
    matW = np.zeros(x_train.shape[1])
    w0_old = np.array([0])
    matW_old = np.zeros(x_train.shape[1])

    # Iterative part
    i = 0
    while not converge and i < max_iteration:
        # Step 1. Update w0
        w0 = np.mean(y_train - x_train @ matW)  # getting mean is equivalent to divided by N

        # Step 2. Find matW
        matW = x_train.T @ x_train
        for k in range(matW.shape[0]): matW[k, k] += lambda_param
        matW = inv(matW) @ x_train.T @ (y_train - w0)

        # Step 3. Finish if weights converge
        diff_w0 = abs(w0 - w0_old)
        diff_matW = np.mean(abs(matW - matW_old))
        converge = True if diff_w0 < threshold and diff_matW < threshold else False

        w0_old = w0
        matW_old = matW
        i += 1

    matW = np.insert(matW, 0, w0, axis=0)
    return matW


def lasso_regression_iterative(x_train:np.ndarray, y_train:np.ndarray, lambda_param:float, threshold:float=0.00001, max_iteration:int=1000):
    """get optimal weights obtained by Lasso Regression.

    Args:
        x_train : ndim array of `[Phi_1, Phi_2, ..., Phi_N]`
        y_train : 1dim array of `[y_1, y_2, ..., y_N]`
        lambda_param : control variable to reduce search region of w
    """
    converge = False

    # Step 0. Give initial values (zeros)
    w0 = np.array([0])
    matW = np.zeros(x_train.shape[1])
    w0_old = np.array([0])
    matW_old = np.zeros(x_train.shape[1])

    mean_x = np.mean(x_train, axis=0)
    mean_y = np.mean(y_train, axis=0)
    X_o = x_train - mean_x
    y_o = y_train - mean_y
    a = np.sum(X_o * X_o, axis=0)

    # Recursive part
    i = 0
    while not converge and i < max_iteration:
        # Step 1. Update w0
        w0 = mean_y - np.sum(mean_x * matW)

        # Step 2. Update matW
        for j in range(matW.shape[0]):
            cj = 0
            for n in range(X_o.shape[0]):
                cj += (y_o[n] - np.sum(X_o[n, :] @ matW) + matW[j] * X_o[n, j]) * X_o[n, j]

            if cj > lambda_param:
                matW[j] = (cj - lambda_param) / a[j]
            elif cj < (-1) * lambda_param:
                matW[j] = (cj + lambda_param) / a[j]
            else:
                matW[j] = 0

        # Step 3. Finish if weights converge
        diff_w0 = abs(w0 - w0_old)
        diff_matW = abs(matW - matW_old).max()
        converge = True if diff_w0 < threshold and diff_matW < threshold else False

        w0_old = w0
        matW_old = matW
        i += 1

    matW = np.insert(matW, 0, w0, axis=0)
    return matW


def least_square_linear_regression(x_train:np.ndarray, y_train:np.ndarray):
    """Least Square linear regression"""
    matX = np.ones((x_train.shape[0], x_train.shape[1]+1))
    matX[:, 1:x_train.shape[1]+1] = DATA_X[:, 0:x_train.shape[1]]
    matW = inv(matX.T @ matX) @ matX.T @ y_train
    return matW


if __name__ == "__main__":
    # Least Square (as a comparison to Ridge & Lasso)
    print(">> Least Square Linear Regression :")
    weights = least_square_linear_regression(DATA_X, DATA_Y)
    print(f"Optimal weights = {weights}")

    # Ridge Regression
    print("\n>> Ridge Regression :")
    for lambda_param in LAMBDA_LIST:
        weights = ridge_regression_iterative(DATA_X, DATA_Y, lambda_param, threshold=0)
        print(f"Optimal weights (lambda={lambda_param}) = {weights}")
        # weights = ridge_regression(DATA_X, DATA_Y, lambda_param)
        # print(f"Optimal weights (lambda={lambda_param}) = {weights} (by formula)")

    # Lasso Regression
    print("\n>> Lasso Regression :")
    for lambda_param in LAMBDA_LIST:
        weights = lasso_regression_iterative(DATA_X, DATA_Y, lambda_param, threshold=0)
        print(f"Optimal weights (lambda={lambda_param}) = {weights}")
