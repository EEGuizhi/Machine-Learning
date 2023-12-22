# KNN & Random Forest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


NUM_OF_NEIGHBORS_LIST = range(1, 101)
NUM_OF_ESTIMATOR_LIST = range(1, 101)
DATA_PATH = "D:/College_Related/Machine_Learning/Assignment_3/column_2C_weka.csv"


if __name__ == "__main__":
    plt.figure(figsize=(10, 8))

    # data reading
    data = pd.read_csv(DATA_PATH)
    data_x = np.stack([
        data["pelvic_incidence"],
        data["pelvic_tilt numeric"],
        data["lumbar_lordosis_angle"],
        data["sacral_slope"],
        data["pelvic_radius"],
        data["degree_spondylolisthesis"]
    ], axis=-1)
    data_y = data["class"]

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=10)


    # KNN
    print(">> KNN algorithm")
    training_result = np.empty(len(NUM_OF_NEIGHBORS_LIST)) 
    testing_result = np.empty(len(NUM_OF_NEIGHBORS_LIST))
    for n_idx in range(len(NUM_OF_NEIGHBORS_LIST)):
        # build model
        KNN_model = KNeighborsClassifier(n_neighbors=NUM_OF_NEIGHBORS_LIST[n_idx])

        # training & testing
        KNN_model.fit(x_train, y_train)
        # print(f"For n_neighbors = {NUM_OF_NEIGHBORS_LIST[n_idx]}:  ", end='')
        # print(f"training acc = {round(KNN_model.score(x_train, y_train), 5)}, testing acc = {round(KNN_model.score(x_test, y_test), 5)}")

        # save result
        training_result[n_idx] = KNN_model.score(x_train, y_train)
        testing_result[n_idx] = KNN_model.score(x_test, y_test)
    print(f"Best pred. n_neighbors = {NUM_OF_NEIGHBORS_LIST[np.argmax(testing_result)]}, Acc = {round(testing_result.max(), 7)*100}%\n")

    plt.subplot(2, 1, 1)
    plt.plot(np.array(NUM_OF_NEIGHBORS_LIST), training_result, label="KNN training acc")
    plt.plot(np.array(NUM_OF_NEIGHBORS_LIST), testing_result, label="KNN testing acc")
    plt.xlabel("number of neighbors"), plt.ylabel("testing accuracy")
    plt.ylim((0, 1))
    plt.legend()


    # Random Forest
    print(">> Random Forest algorithm")
    testing_result = np.empty(len(NUM_OF_ESTIMATOR_LIST))
    for n_idx in range(len(NUM_OF_ESTIMATOR_LIST)):
        # build model
        RF_model = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATOR_LIST[n_idx], criterion="gini", max_depth=3)

        # training & testing
        RF_model.fit(x_train, y_train)
        # print(f"For n_estimators = {NUM_OF_ESTIMATOR_LIST[n_idx]}:  ", end='')
        # print(f"training acc = {round(RF_model.score(x_train, y_train), 5)}, testing acc = {round(RF_model.score(x_test, y_test), 5)}")

        # save result
        training_result[n_idx] = RF_model.score(x_train, y_train)
        testing_result[n_idx] = RF_model.score(x_test, y_test)
    print(f"Best pred. n_estimators = {NUM_OF_ESTIMATOR_LIST[np.argmax(testing_result)]}, Acc = {round(testing_result.max(), 7)*100}%\n")
    
    plt.subplot(2, 1, 2)
    plt.plot(np.array(NUM_OF_ESTIMATOR_LIST), training_result, label="RF training acc")
    plt.plot(np.array(NUM_OF_ESTIMATOR_LIST), testing_result, label="RF testing acc")
    plt.xlabel("number of estimators"), plt.ylabel("testing accuracy")
    plt.ylim((0, 1))
    plt.legend()

    plt.show()
