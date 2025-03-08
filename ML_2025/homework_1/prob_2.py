# BSChen
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH = "D:/NYCU/Machine_learning/TA/homework_1/img041-019.png"
# PATH = "D:/NYCU/Machine_learning/TA/homework_1/img011-040.png"
# N_BASIS = 6
N_BASIS = 10

def linear_regression(mat_X: np.ndarray, mat_Y: np.ndarray) -> np.ndarray:
    """Linear Regression"""
    mat_W = np.linalg.inv(mat_X.T @ mat_X) @ mat_X.T @ mat_Y
    return mat_W

def sin_cos_basis(train_X: np.ndarray, n_basis: int) -> np.ndarray:
    """Inputs through sine & cosine basis functions"""
    mat_X = np.ones((train_X.shape[0], n_basis**2 * 2 + 1))
    for i in range(n_basis):
        for j in range(n_basis):
            mat_X[:, 2 * (i * n_basis + j) + 1] = np.sin(
                (i + 1) * (2 * np.pi) * (train_X[:, 0] - 0.5) +
                (j + 1) * (2 * np.pi) * (train_X[:, 1] - 0.5)
            )
            mat_X[:, 2 * (i * n_basis + j) + 2] = np.cos(
                (i + 1) * (2 * np.pi) * (train_X[:, 0] - 0.5) +
                (j + 1) * (2 * np.pi) * (train_X[:, 1] - 0.5)
            )
    return mat_X

def polynomial_basis(train_X: np.ndarray, n_basis: int) -> np.ndarray:
    """Inputs through polynomial basis functions"""
    mat_X = np.ones((train_X.shape[0], n_basis**2 + 1))
    for i in range(n_basis):
        for j in range(n_basis):
            mat_X[:, i * n_basis + j + 1] = train_X[:, 0]**(i + 1) * train_X[:, 1]**(j + 1)
    return mat_X

def gaussian_basis(train_X: np.ndarray, n_basis: int) -> np.ndarray:
    """Inputs through Gaussian basis functions"""
    mat_X = np.ones((train_X.shape[0], n_basis**2 + 1))
    var = 1 / n_basis
    for i in range(n_basis):
        for j in range(n_basis):
            mat_X[:, i * n_basis + j + 1] = np.exp(
                -((train_X[:, 0] - (i + 1) / n_basis)**2 +
                  (train_X[:, 1] - (j + 1) / n_basis)**2) / (2 * var**2)
            )
    return mat_X

def MAE(mat_Y: np.ndarray, mat_Y_hat: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(mat_Y - mat_Y_hat))

def MSE(mat_Y: np.ndarray, mat_Y_hat: np.ndarray) -> float:
    """Mean Squared Error"""
    return np.mean((mat_Y - mat_Y_hat)**2)

if __name__ == "__main__":
    # Make outputs of training data
    data = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    data = 1 - data / 255
    train_Y = data.flatten()

    # Make inputs of training data
    col, row = np.meshgrid(
        np.linspace(0, 1, num=data.shape[1]),  # x-axis
        np.linspace(0, 1, num=data.shape[0]),  # y-axis
        indexing="xy"
    )
    train_X = np.stack((row.flatten(), col.flatten()), axis=-1)

    # Inputs basis functions
    # mat_X = sin_cos_basis(train_X, N_BASIS)
    # mat_X = polynomial_basis(train_X, 6)
    mat_X = gaussian_basis(train_X, N_BASIS)

    # Linear Regression
    mat_W = linear_regression(mat_X, train_Y)
    print(f"weights shape: {mat_W.shape}")
    train_Y_hat = mat_X @ mat_W
    print(f"MAE: {MAE(train_Y, train_Y_hat)}, MSE: {MSE(train_Y, train_Y_hat)}")

    # Plot
    train_Y_hat[train_Y_hat < 0] = 0
    train_Y_hat[train_Y_hat > 1] = 1
    train_Y_hat = train_Y_hat.reshape(data.shape)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(data, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(train_Y_hat, cmap="gray")
    plt.show()
