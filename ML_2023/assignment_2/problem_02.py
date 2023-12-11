# Ridge Regression
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

SAMPLES = 30
SIGMA_LIST = (0.01, 0.1, 1, 2)
LAMBDA_LIST = (0, 0.01, 0.1, 1, 10)

def truth_func(x:float):
    return 3 + 2*x + 0.2*(x**2)

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


if __name__ == "__main__":
    plt.figure(figsize=(12, 9))
    for sigma_idx in range(len(SIGMA_LIST)):
        # Generate Dataset
        dataset = np.empty((SAMPLES, 2))
        for i in range(SAMPLES):
            x = np.random.uniform(0, 8, size=None)
            y = truth_func(x) + np.random.normal(0, SIGMA_LIST[sigma_idx], size=None)
            dataset[i] = x, y
        mat_X = np.empty((SAMPLES, 2))
        mat_X[:, 0] = dataset[:, 0]
        mat_X[:, 1] = dataset[:, 0] * dataset[:, 0]
        mat_Y = np.empty((SAMPLES, 1))
        mat_Y[:, 0] = dataset[:, 1]

        print(f"Regression {sigma_idx+1} (sigma = {SIGMA_LIST[sigma_idx]}):")
        for lambda_param in LAMBDA_LIST:
            # Ridge Regression
            mat_W = ridge_regression(mat_X, mat_Y, lambda_param)

            # Plot
            x = np.linspace(0, 8, num=200)
            y = mat_W[2] * x**2 + mat_W[1] * x + mat_W[0]
            print(f"y = {round(mat_W[0].item(), 3)} + {round(mat_W[1].item(), 3)}*x + {round(mat_W[2].item(), 3)}*x^2  (lambda = {lambda_param})")
            plt.subplot(2, 2, sigma_idx+1)
            plt.plot(x, y, label=f"lambda = {lambda_param}")

        plt.title(f"Regression {sigma_idx+1} (sigma = {SIGMA_LIST[sigma_idx]})")
        plt.plot(dataset[:, 0], dataset[:, 1], "ro", markersize=2, label="data points")
        plt.legend(loc="upper left")
        print("")
    plt.show()
