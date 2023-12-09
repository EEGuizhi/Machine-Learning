# Ridge & Lasso Regression
import numpy as np
from numpy.linalg import inv

LAMBDA_LIST = (0.01, 0.1, 1, 10)
DATA_X = np.array([  # N = dim 0, M = dim 1
    (1, 1),
    (1, 3),
    (5, 2),
    (7, 4),
    (9, 2)
])
DATA_Y = np.array([
    2,
    5,
    3,
    4,
    8
])


def ridge_regression(x_train:np.ndarray, y_train:np.ndarray, lambda_:float):
    """get optimal weights obtained by Ridge Regression.

    Args:
        x_train : ndim array of `[Phi_1, Phi_2, ..., Phi_N]`
        y_train : 1dim array of `[y_1, y_2, ..., y_N]`
        lambda_ : control variable to reduce search region of w
    """

    mean_x = np.mean(x_train, axis=0)
    mean_y = np.mean(y_train, axis=0)
    X_o = x_train - mean_x
    y_o = y_train - mean_y

    matW = X_o.T @ X_o
    for i in range(matW.shape[0]): matW[i, i] += lambda_
    matW = inv(matW) @ X_o.T @ y_o
    w0 = mean_y - mean_x @ matW

    matW = np.insert(matW, 0, w0, axis=0)
    return matW


def ridge_regression_recursive(x_train:np.ndarray, y_train:np.ndarray, lambda_:float, threshold:float=0.0001, max_iteration:int=1000):
    """same as the ridge_regression() function, but this is recursive.

    Args:
        x_train : ndim array of `[Phi_1, Phi_2, ..., Phi_N]`
        y_train : 1dim array of `[y_1, y_2, ..., y_N]`
        lambda_ : control variable to reduce search region of w
    """
    converge = False

    # Step 0. Give initial values (zeros)
    w0 = np.array([0])
    matW = np.zeros(x_train.shape[-1])
    w0_old = np.array([0])
    matW_old = np.zeros(x_train.shape[-1])

    # Recursive part
    i = 0
    while not converge and i < max_iteration:
        # Step 1. Update w0
        w0 = y_train - x_train @ matW
        w0 = np.mean(w0)  # getting mean is equivalent to divided by N

        # Step 2. Find matW
        matW = x_train.T @ x_train
        for i in range(matW.shape[0]): matW[i, i] += lambda_
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


def lasso_regression(x_train:np.ndarray, y_train:np.ndarray, lambda_:float, threshold:float=0.0001, max_iteration:int=1000):
    """get optimal weights obtained by Lasso Regression.

    Args:
        x_train : ndim array of `[Phi_1, Phi_2, ..., Phi_N]`
        y_train : 1dim array of `[y_1, y_2, ..., y_N]`
        lambda_ : control variable to reduce search region of w
    """
    converge = False

    # Step 0. Give initial values (zeros)
    w0 = np.array([0])
    matW = np.zeros(x_train.shape[-1])
    w0_old = np.array([0])
    matW_old = np.zeros(x_train.shape[-1])
    mean_x = np.mean(x_train, axis=0)

    # Recursive part
    i = 0
    while not converge and i < max_iteration:
        # Step 1. Update w0
        w0 = np.mean(y_train) + np.mean(mean_x @ matW)  # getting mean is equivalent to divided by N

        # Step 2. Update matW
        matW = x_train.T @ x_train
        for i in range(matW.shape[0]): matW[i, i] += lambda_
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




if __name__ == "__main__":
    # Ridge Regression
    print(">> Ridge Regression :")
    for lambda_idx in range(len(LAMBDA_LIST)):
        weights = ridge_regression(DATA_X, DATA_Y, LAMBDA_LIST[lambda_idx])
        print(f"Optimal weights (lambda={LAMBDA_LIST[lambda_idx]}) = {weights} (by direct calc)")

        weights = ridge_regression_recursive(DATA_X, DATA_Y, LAMBDA_LIST[lambda_idx])
        print(f"Optimal weights (lambda={LAMBDA_LIST[lambda_idx]}) = {weights}\n")

    # Lasso Regression
    print(">> Lasso Regression :")
    for lambda_idx in range(len(LAMBDA_LIST)):
        weights = ridge_regression_recursive(DATA_X, DATA_Y, LAMBDA_LIST[lambda_idx])
        print(f"Optimal weights (lambda={LAMBDA_LIST[lambda_idx]}) = {weights}")
