# Lasso Regression
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


SAMPLES = 30
SIGMA_LIST = (0.01, 0.1, 1, 2)
LAMBDA_LIST = (0, 0.01, 0.1, 1, 10)


def truth_func(x:float):
    return 3 + 2*x + 0.2*(x**2)


def lasso_regression(x_train:np.ndarray, y_train:np.ndarray, lambda_param:float):
    """This function has bug at sometimes, but also can get optimal weights obtained by Lasso Regression.

    Args:
        x_train : ndim array of `[Phi_1, Phi_2, ..., Phi_N]`
        y_train : 1dim array of `[y_1, y_2, ..., y_N]`
        lambda_param : control variable to reduce search region of w
    """
    mean_x = np.mean(x_train, axis=0)
    mean_y = np.mean(y_train, axis=0)
    X_o = x_train - mean_x
    y_o = y_train - mean_y

    weight_is_pos = np.ones(x_train.shape[1])
    while True:
        # calc
        matW = inv(X_o.T @ X_o)
        matW_pos = X_o.T @ y_o
        for i in range(matW_pos.shape[0]):
            matW_pos[i] += (-1)*lambda_param if weight_is_pos[i] > 0 else lambda_param
        matW = matW @ matW_pos

        # check
        flag = True
        for i in range(matW_pos.shape[0]):
            if not matW[i] * weight_is_pos[i] >= 0:
                flag = False
                weight_is_pos[i] = -1
        if flag:
            break

    w0 = mean_y - mean_x @ matW

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

    # Iterative part
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

        print(f"Lasso Regression {sigma_idx+1} (sigma = {SIGMA_LIST[sigma_idx]}):")
        for lambda_param in LAMBDA_LIST:
            # Lasso Regression
            mat_W = lasso_regression_iterative(mat_X, mat_Y, lambda_param)

            # Plot
            x = np.linspace(0, 8, num=200)
            y = mat_W[2] * x**2 + mat_W[1] * x + mat_W[0]
            print(f"y = {round(mat_W[0].item(), 3)} + {round(mat_W[1].item(), 3)}*x + {round(mat_W[2].item(), 3)}*x^2  (lambda = {lambda_param})")
            plt.subplot(2, 2, sigma_idx+1)
            plt.plot(x, y, label=f"lambda = {lambda_param}")

        plt.title(f"Lasso Regression {sigma_idx+1} (sigma = {SIGMA_LIST[sigma_idx]})")
        plt.plot(dataset[:, 0], dataset[:, 1], "ro", markersize=2, label="data points")
        plt.legend(loc="upper left")
        print("")
    plt.show()
