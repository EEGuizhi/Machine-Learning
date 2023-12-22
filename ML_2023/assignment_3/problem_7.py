# Gaussian Basis Function for Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

N_SAMPLE = 100
NUM_OF_TERMS = 5
SIGMA_VALUES = [0.1, 0.5, 1.0, 2.0]


# Gaussian basis function
def get_gaussian_features(x:np.ndarray, n_terms:int, sigma:float):
    features = np.ones((x.shape[0], 1))
    for j in range(1, n_terms + 1):
        mu_j = -2 + 4 * (j - 1) / n_terms
        features = np.concatenate((features, np.expand_dims(np.exp(-(x - mu_j) ** 2 / (2 * sigma ** 2)), axis=0).T), axis=1)
    return features


if __name__ == "__main__":
    # generate dataset
    random_gen = np.random.default_rng()
    x = random_gen.uniform(-2, 2, N_SAMPLE)
    y = np.sin(4 * x) + 0.1 * random_gen.normal(0, 1, N_SAMPLE)

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    plt.xlabel("Feature"), plt.ylabel("Target")

    # Linear Regression with Gaussian Basis Function
    for sigma in SIGMA_VALUES:
        model = LinearRegression()

        # training & testing
        x_features = get_gaussian_features(x, NUM_OF_TERMS, sigma)
        model.fit(x_features, y)

        # plot result
        x_samples = np.linspace(-2, 2, num=200)
        pred_x = get_gaussian_features(x_samples, NUM_OF_TERMS, sigma)
        pred_y = model.predict(pred_x)
        plt.plot(x_samples, pred_y, label=f"Sigma = {sigma}")

    plt.legend()
    plt.xlim((-2, 2))
    plt.show()
