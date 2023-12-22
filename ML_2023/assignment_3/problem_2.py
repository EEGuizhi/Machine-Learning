# Decision Tree + Wine dataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


MAX_DEPTH_LIST = range(1, 13)


if __name__ == "__main__":
    # data reading
    wine_data = load_wine()

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.25, random_state=10)

    # Decision tree
    testing_result = np.empty(len(MAX_DEPTH_LIST))
    for n_idx in range(len(MAX_DEPTH_LIST)):
        print(f"For max_depth = {MAX_DEPTH_LIST[n_idx]}:  ", end='')
        
        # build model
        DT_model = DecisionTreeClassifier(criterion="gini", max_depth=MAX_DEPTH_LIST[n_idx])
        
        # training & testing
        DT_model.fit(x_train, y_train)
        print(f"training acc = {round(DT_model.score(x_train, y_train), 5)}, testing acc = {round(DT_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = DT_model.score(x_test, y_test)
    print(f"Best pred. max_depth = {MAX_DEPTH_LIST[np.argmax(testing_result)]}\n")

    # build best model
    DT_model = DecisionTreeClassifier(criterion="gini", max_depth=MAX_DEPTH_LIST[np.argmax(testing_result)])
    DT_model.fit(x_train, y_train)
    
    # show importance
    x_idx = np.arange(len(wine_data.feature_names))
    plt.barh(x_idx, DT_model.feature_importances_)
    plt.yticks(x_idx, wine_data.feature_names)
    plt.show()
