# Logistic Regression
import numpy as np
import cvxopt
import matplotlib.pyplot as plt


DATA_X = np.array([  # inputs
    [2, 3],
    [4, 2],
    [1, 1],
    [2, 0]
])

DATA_Y = np.array([  # labels
    0,
    0,
    1,
    1
])


class logistic_regression:
    def __init__(self, learning_rate:float=0.001, epoch:int=10000):
        self.lr = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None
    
    def sigmoid(self, x:np.ndarray):
        return 1 / (1 + np.exp(-x))

    def train(self, x_train:np.ndarray, y_train:np.ndarray):
        # init
        bias_term = np.ones(x_train.shape[0])  # insert bias term into weights
        x_train = np.append(x_train, np.expand_dims(bias_term, axis=1), axis=1)
        self.weights = np.zeros(x_train.shape[1])

        # gradient descent update weights & bias
        for ep in range(self.epoch):
            pred = self.sigmoid(x_train @ self.weights)
            loss = y_train - pred  # coef
            grad = np.sum(
                np.repeat(np.expand_dims(loss, axis=1), x_train.shape[1], axis=1) * x_train,
                axis=0
            )
            self.weights += self.lr * grad

        # separate bias term from weights
        self.bias = self.weights[-1]
        self.weights = self.weights[0:-1]

    def predict(self, x:np.ndarray):
        y = self.sigmoid(self.weights.T @ x + self.bias)
        return 1 if y > 0.5 else 0


if __name__ == "__main__":
    cvxopt.solvers.options['show_progress'] = False  # silent

    # Logistic Regression
    logistic_model = logistic_regression()
    logistic_model.train(DATA_X, DATA_Y)
    print(f">> weights = {logistic_model.weights}, bias = {logistic_model.bias}")

    # Plot
    x = np.array([0, 5])  # pred = sigmoid( w[1]*y + w[0]*x + b ).
    y = -(logistic_model.weights[0] * x + logistic_model.bias) / logistic_model.weights[1]  # when pred = 0.5, w[1]*y + w[0]*x + b = 0

    plt.figure(figsize=(10, 8))
    plt.grid()
    plt.scatter(DATA_X[DATA_Y == 0, 0], DATA_X[DATA_Y == 0, 1], c='red', s=30, label='class 1')
    plt.scatter(DATA_X[DATA_Y == 1, 0], DATA_X[DATA_Y == 1, 1], c='blue', s=30, label='class 2')
    plt.plot(x, y, c='gold', label='decision boundary')

    plt.xlim((0.9, 4.1))
    plt.show()
