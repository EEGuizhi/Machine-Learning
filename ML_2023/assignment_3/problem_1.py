# KNN & Random Forest + custom dataset
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


NUM_OF_NEIGHBORS_LIST = range(2, 21)
NUM_OF_ESTIMATOR_LIST = range(2, 21)
DATA_PATH = "D:/College_Related/Machine_Learning/Assignment_3/column_2C_weka.csv"


if __name__ == "__main__":
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
    testing_result = np.empty(len(NUM_OF_NEIGHBORS_LIST))
    for n_idx in range(len(NUM_OF_NEIGHBORS_LIST)):
        print(f"For n_neighbors = {NUM_OF_NEIGHBORS_LIST[n_idx]}:  ", end='')
        
        # build model
        KNN_model = KNeighborsClassifier(n_neighbors=NUM_OF_NEIGHBORS_LIST[n_idx])
        
        # training & testing
        KNN_model.fit(x_train, y_train)
        print(f"training acc = {round(KNN_model.score(x_train, y_train), 5)}, testing acc = {round(KNN_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = KNN_model.score(x_test, y_test)
    print(f"Best pred. n_neighbors = {NUM_OF_NEIGHBORS_LIST[np.argmax(testing_result)]}\n")


    # Random Forest
    print(">> Random Forest algorithm")
    testing_result = np.empty(len(NUM_OF_ESTIMATOR_LIST))
    for n_idx in range(len(NUM_OF_ESTIMATOR_LIST)):
        print(f"For n_estimators = {NUM_OF_ESTIMATOR_LIST[n_idx]}:  ", end='')
        
        # build model
        RF_model = RandomForestClassifier(n_estimators=NUM_OF_ESTIMATOR_LIST[n_idx], criterion="gini", max_depth=3)
        
        # training & testing
        RF_model.fit(x_train, y_train)
        print(f"training acc = {round(RF_model.score(x_train, y_train), 5)}, testing acc = {round(RF_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = RF_model.score(x_test, y_test)
    print(f"Best pred. n_estimators = {NUM_OF_ESTIMATOR_LIST[np.argmax(testing_result)]}\n")
