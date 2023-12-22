# Random Forest + Digits Dataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


NUM_OF_ESTIMATOR_LIST = range(1, 101)


if __name__ == "__main__":
    # data reading
    digits_data = load_digits()

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(digits_data.data, digits_data.target, test_size=0.25, random_state=10)

    # save result
    highest_acc = 0
    feature_importance = None

    # Random Forest
    print(">> Random Forest algorithm")
    training_result = np.empty(len(NUM_OF_ESTIMATOR_LIST))
    testing_result = np.empty(len(NUM_OF_ESTIMATOR_LIST))
    for n_idx in range(len(NUM_OF_ESTIMATOR_LIST)):
        # build model
        RF_model = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATOR_LIST[n_idx], criterion="gini", max_depth=10)

        # training & testing
        RF_model.fit(x_train, y_train)
        # print(f"For n_estimators = {NUM_OF_ESTIMATOR_LIST[n_idx]}:  ", end='')
        # print(f"training acc = {round(RF_model.score(x_train, y_train), 5)}, testing acc = {round(RF_model.score(x_test, y_test), 5)}")

        # save result
        training_result[n_idx] = RF_model.score(x_train, y_train)
        testing_result[n_idx] = RF_model.score(x_test, y_test)
        if testing_result[n_idx] > highest_acc:
            feature_importance = RF_model.feature_importances_
            highest_acc = testing_result[n_idx]
    print(f"Best pred. n_estimators = {NUM_OF_ESTIMATOR_LIST[np.argmax(testing_result)]}, Acc = {round(testing_result.max(), 7)*100}%\n")


    # plot acc
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(NUM_OF_ESTIMATOR_LIST), training_result, label="training acc")
    plt.plot(np.array(NUM_OF_ESTIMATOR_LIST), testing_result, label="testing acc")
    plt.xlabel("number of estimators"), plt.ylabel("testing accuracy")
    plt.ylim((0, 1.01))
    plt.legend()

    # plot importance
    plt.figure(figsize=(16, 12))
    feature_names = [f"pixel({i // 8}, {i % 8})" for i in range(64)]  # y, x
    x_idx = np.arange(digits_data.data.shape[1])
    plt.barh(x_idx, feature_importance)
    plt.yticks(x_idx, feature_names)
    plt.xlabel("feature importance")

    plt.show()
