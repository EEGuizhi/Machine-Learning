# Logistic Regression for Mutiple Classes
import numpy as np
import cvxopt
import matplotlib.pyplot as plt


DATA_POINTS_EACH_CLASS = 50


class logistic_multiclass_regression:
    def __init__(self, learning_rate:float=0.001, epoch:int=10000):
        self.lr = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def softmax(self, x:np.ndarray) -> np.ndarray:  # for multiple samples prediction
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), axis=1).repeat(x.shape[1], axis=1)

    def train(self, x_train:np.ndarray, y_train:np.ndarray):
        # init
        num_class = y_train.shape[1]
        dim_input = x_train.shape[1]
        bias_term = np.ones(x_train.shape[0])  # insert bias term into weights
        x_train = np.append(x_train, np.expand_dims(bias_term, axis=1), axis=1)
        self.weights = np.zeros((dim_input+1, num_class))

        # gradient descent update weights & bias
        for ep in range(self.epoch):
            pred = self.softmax(x_train @ self.weights)
            loss = y_train - pred  # coef
            grad = x_train.T @ loss
            self.weights += self.lr * grad

        # separate bias term from weights
        self.bias = self.weights[-1]
        self.weights = self.weights[0:-1]

    def predict(self, x:np.ndarray):
        if len(x.shape) == 1:
            y = self.softmax(np.expand_dims(x @ self.weights + self.bias, axis=0))
        else:
            y = self.softmax(x @ self.weights + self.bias)
        return np.argmax(y, axis=1)[0] if len(x.shape) == 1 else np.argmax(y, axis=1)


if __name__ == "__main__":
    cvxopt.solvers.options['show_progress'] = False  # silent

    # Dataset
    cov = np.array([
        [1, 0],
        [0, 1]
    ])
    mean = np.array([1, 1])
    data_x = np.random.multivariate_normal(mean, cov, DATA_POINTS_EACH_CLASS)
    mean = np.array([-3, -3])
    data_x = np.append(data_x, np.random.multivariate_normal(mean, cov, DATA_POINTS_EACH_CLASS), axis=0)
    mean = np.array([-6, 2])
    data_x = np.append(data_x, np.random.multivariate_normal(mean, cov, DATA_POINTS_EACH_CLASS), axis=0)
    data_y = np.concatenate((
        np.tile([[1, 0, 0]], (DATA_POINTS_EACH_CLASS, 1)),
        np.tile([[0, 1, 0]], (DATA_POINTS_EACH_CLASS, 1)),
        np.tile([[0, 0, 1]], (DATA_POINTS_EACH_CLASS, 1))
    ), axis=0)

    # Logistic Regression
    logistic_model = logistic_multiclass_regression()
    logistic_model.train(data_x, data_y)
    print(f">> weights = \n{logistic_model.weights}")
    print(f">> bias = {logistic_model.bias}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.grid()
    plt.scatter(data_x[data_y[:, 0] == 1, 0], data_x[data_y[:, 0] == 1, 1], c='red', s=30, label='class 1')
    plt.scatter(data_x[data_y[:, 1] == 1, 0], data_x[data_y[:, 1] == 1, 1], c='orange', s=30, label='class 2')
    plt.scatter(data_x[data_y[:, 2] == 1, 0], data_x[data_y[:, 2] == 1, 1], c='blue', s=30, label='class 3')

    x = np.linspace(np.min(data_x[:, 0]), np.max(data_x[:, 0]), 200)
    y = np.linspace(np.min(data_x[:, 1]), np.max(data_x[:, 1]), 200)
    xx, yy = np.meshgrid(x, y)
    x_in = np.c_[xx.ravel(), yy.ravel()]

    pred = logistic_model.predict(x_in).reshape(xx.shape)
    plt.contourf(xx, yy, pred, cmap=plt.cm.RdYlBu, alpha=0.5)

    plt.legend()
    plt.show()
