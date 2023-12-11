# Linear Regression (Least Square)
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

SIGMA_LIST = [0.01, 0.1, 1, 2]
SAMPLES = 40

def linear_regression(mat_X:np.ndarray, mat_Y:np.ndarray):
    # Linear Regression
    mat_W = inv(mat_X.T @ mat_X) @ mat_X.T @ mat_Y
    return mat_W


if __name__ == "__main__":
    # Problem.1 ~ Problem.4
    for sigma_idx in range(len(SIGMA_LIST)):

        # Generate Dataset
        dataset = np.empty((SAMPLES, 2))
        for i in range(SAMPLES):
            x = np.random.uniform(0, 8, size=None)
            y = 3 + 2*x + 0.2*(x**2) + np.random.normal(0, SIGMA_LIST[sigma_idx], size=None)
            dataset[i] = x, y

        # Linear Regression
        mat_X = np.ones((SAMPLES, 3))
        mat_X[:, 1] = dataset[:, 0]
        mat_X[:, 2] = dataset[:, 0] * dataset[:, 0]
        mat_Y = np.empty((SAMPLES, 1))
        mat_Y[:, 0] = dataset[:, 1]
        mat_W = linear_regression(mat_X, mat_Y)

        # Plot
        x = np.linspace(0, 8, num=200)
        y = mat_W[2] * x**2 + mat_W[1] * x + mat_W[0]
        print(f"Regression {sigma_idx+1} (sigma = {SIGMA_LIST[sigma_idx]}):")
        print(f"y = {round(mat_W[2].item(), 3)} * x^2  +  {round(mat_W[1].item(), 3)} * x  +  {round(mat_W[0].item(), 3)} \n")
        plt.subplot(2, 2, sigma_idx+1)
        plt.title(f"Regression {sigma_idx+1} (sigma = {SIGMA_LIST[sigma_idx]})")
        plt.plot(dataset[:, 0], dataset[:, 1], "ro", markersize=2, label="dataset")
        plt.plot(x, y, label="model")
        plt.legend(loc="upper left")
