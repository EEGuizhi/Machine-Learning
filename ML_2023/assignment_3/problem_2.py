# Decision Tree
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


MAX_DEPTH_LIST = range(1, 101)


if __name__ == "__main__":
    plt.figure(figsize=(10, 8))

    # data reading
    wine_data = load_wine()

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.25, random_state=10)

    # save result
    highest_acc = 0
    feature_importance = None

    # Decision tree
    training_result = np.empty(len(MAX_DEPTH_LIST))
    testing_result = np.empty(len(MAX_DEPTH_LIST))
    for n_idx in range(len(MAX_DEPTH_LIST)):
        # build model
        DT_model = DecisionTreeClassifier(criterion="gini", max_depth=MAX_DEPTH_LIST[n_idx])

        # training & testing
        DT_model.fit(x_train, y_train)
        # print(f"For max_depth = {MAX_DEPTH_LIST[n_idx]}:  ", end='')
        # print(f"training acc = {round(DT_model.score(x_train, y_train), 5)}, testing acc = {round(DT_model.score(x_test, y_test), 5)}")

        # save result
        training_result[n_idx] = DT_model.score(x_train, y_train)
        testing_result[n_idx] = DT_model.score(x_test, y_test)
        if testing_result[n_idx] > highest_acc:
            feature_importance = DT_model.feature_importances_
            highest_acc = testing_result[n_idx]
    print(f"Best pred. max_depth = {MAX_DEPTH_LIST[np.argmax(testing_result)]}, Acc = {round(testing_result.max(), 7)*100}%\n")

    # plot acc
    plt.subplot(2, 1, 1)
    plt.plot(np.array(MAX_DEPTH_LIST), training_result, label="training acc")
    plt.plot(np.array(MAX_DEPTH_LIST), testing_result, label="testing acc")
    plt.xlabel("max depth"), plt.ylabel("testing accuracy")
    plt.ylim((0, 1.01))
    plt.legend()

    # plot importance
    plt.subplot(2, 1, 2)
    x_idx = np.arange(len(wine_data.feature_names))
    plt.barh(x_idx, feature_importance)
    plt.yticks(x_idx, wine_data.feature_names)
    plt.xlabel("feature importance")

    plt.show()
