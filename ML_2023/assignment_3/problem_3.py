# Random Forest + Digits Dataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


NUM_OF_ESTIMATOR_LIST = range(2, 30, 2)


if __name__ == "__main__":
    # data reading
    digits_data = load_digits()

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(digits_data.data, digits_data.target, test_size=0.25, random_state=10)

    # Random Forest
    print(">> Random Forest algorithm")
    testing_result = np.empty(len(NUM_OF_ESTIMATOR_LIST))
    for n_idx in range(len(NUM_OF_ESTIMATOR_LIST)):
        print(f"For n_estimators = {NUM_OF_ESTIMATOR_LIST[n_idx]}:  ", end='')
        
        # build model
        RF_model = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATOR_LIST[n_idx], criterion="gini", max_depth=10)
        
        # training & testing
        RF_model.fit(x_train, y_train)
        print(f"training acc = {round(RF_model.score(x_train, y_train), 5)}, testing acc = {round(RF_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = RF_model.score(x_test, y_test)
    print(f"Best pred. n_estimators = {NUM_OF_ESTIMATOR_LIST[np.argmax(testing_result)]}\n")


    # build best model
    RF_model = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATOR_LIST[np.argmax(testing_result)], criterion="gini", max_depth=10)
    RF_model.fit(x_train, y_train)

    # show importance
    plt.figure(figsize=(10, 12))
    feature_names = [f"pixel({i // 8}, {i % 8})" for i in range(64)]  # y, x
    x_idx = np.arange(digits_data.data.shape[1])
    plt.barh(x_idx, RF_model.feature_importances_)
    plt.yticks(x_idx, feature_names)
    plt.show()
