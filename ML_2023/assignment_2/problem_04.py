# Support Vector Machine
import numpy as np
import cvxopt
from cvxopt import matrix
import matplotlib.pyplot as plt


DATA_X = np.array([
    [2, 3],
    [4, 2],
    [1, 1],
    [2, 0]
])

DATA_Y = np.array([
    1,
    1,
    -1,
    -1
])


class svm:
    def __init__(self):
        self.weights = None  # mat_W
        self.bias = None  # w0

    def train(self, x_train:np.ndarray, y_train:np.ndarray):
        n_samples = x_train.shape[0]

        # quadratic programming optimization
        z = np.empty_like(x_train)
        for n in range(n_samples):
            z[n] = x_train[n] * y_train[n]
        matP = (z @ z.T) / 2
        q = np.ones(n_samples) * (-1)
        matG = np.identity(n_samples) * (-1)
        h = np.zeros(n_samples)
        matA = y_train
        b = 0.0

        sol = cvxopt.solvers.qp(
            P = matrix(matP.astype('double')), q = matrix(q.astype('double')),
            G = matrix(matG.astype('double')), h = matrix(h.astype('double')),
            A = matrix(matA.astype('double'), (1, n_samples)), b=matrix(b)
        )
        alpha = np.array(sol['x']).squeeze(axis=-1)

        # update weights & bias
        self.weights = ((alpha * y_train).T @ x_train).T
        sv_idx = np.argsort(alpha)[n_samples - 2: n_samples]  # get index of support vectors
        scale = (y_train[sv_idx[0]] - y_train[sv_idx[1]]) / (self.weights.T @ (x_train[sv_idx[0]] - x_train[sv_idx[1]]))
        self.weights *= scale
        self.bias = y_train[sv_idx[0]] - self.weights.T @ x_train[sv_idx[0]]

    def predict(self, x:np.ndarray):
        y = self.weights.T @ x + self.bias
        if y > 0:
            return 1
        elif y < 0:
            return -1
        else:
            return 0


if __name__ == "__main__":
    cvxopt.solvers.options['show_progress'] = False  # silent

    # Support Vector Machine
    svm_model = svm()
    svm_model.train(DATA_X, DATA_Y)
    print(f">> weights = {svm_model.weights}, bias = {svm_model.bias}")

    # Plot
    x = np.array([0, 5])
    y = -(svm_model.weights[0] * x + svm_model.bias) / svm_model.weights[1]

    plt.figure(figsize=(10, 8))
    plt.grid()
    plt.scatter(DATA_X[DATA_Y == 1, 0], DATA_X[DATA_Y == 1, 1], c='red', s=30, label='class 1')
    plt.scatter(DATA_X[DATA_Y == -1, 0], DATA_X[DATA_Y == -1, 1], c='blue', s=30, label='class 2')
    plt.plot(x, y, c='gold', label='decision boundary')

    plt.xlim((0.9, 4.1))
    plt.show()
