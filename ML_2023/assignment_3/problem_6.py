# Linear Regression + Polynomial basis function(sin function)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


N_SAMPLE = 100
NUM_OF_TERMS_LIST = [1, 3, 5, 7, 9]


def get_polynomial_features(x:np.ndarray, n_terms:int):
    features = np.ones((x.shape[0], 1))
    for i in range(n_terms - 1):
        features = np.concatenate((features, np.expand_dims(np.power(x, i+1), axis=0).T), axis=1)
    return features


if __name__ == "__main__":
    # generate dataset
    random_gen = np.random.default_rng()
    x = random_gen.uniform(-2, 2, N_SAMPLE)
    y = np.sin(4 * x) + 0.1*random_gen.normal(0, 1, N_SAMPLE)

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)
    plt.xlabel("Feature"), plt.ylabel("Target")

    # Linear Regression
    for n_terms in NUM_OF_TERMS_LIST:
        model = LinearRegression()

        # training & testing
        x_features = get_polynomial_features(x, n_terms)
        model.fit(x_features, y)
        
        # plot result
        x_samples = np.linspace(-2, 2, num=200)
        pred_x = get_polynomial_features(x_samples, n_terms)
        pred_y = model.predict(pred_x)
        plt.plot(x_samples, pred_y, label=f"num of terms = {n_terms}")
    plt.legend()
    plt.xlim((-2, 2))
    plt.show()
