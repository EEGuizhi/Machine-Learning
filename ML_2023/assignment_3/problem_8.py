# Gaussian Basis Function for Linear Regression
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


REGULARIZE_PARAM_LIST = [0.001, 0.01, 0.1, 1, 10, 100]
NUM_OF_ESTIMATOR_LIST = range(2, 11, 2)
DATA_PATH = "/content/drive/MyDrive/Shared_Things/Machine_Learning/Assignments/hw3/pima-indians-diabetes.csv"


if __name__ == "__main__":
    # data reading
    data = pd.read_csv(DATA_PATH)
    data_x = np.stack([
        data["Pregnancies"],
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["DiabetesPedigreeFunction"],
        data["Age"]
    ], axis=-1)
    data_y = data["Outcome"]

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=10)


    # Logistic Regression
    print(">> Logistic Regression")
    testing_result = np.empty(len(REGULARIZE_PARAM_LIST))
    for n_idx in range(len(REGULARIZE_PARAM_LIST)):
        print(f"For C = {REGULARIZE_PARAM_LIST[n_idx]}:  ", end='')
        
        # build model
        Logistic_model = LogisticRegression(C=REGULARIZE_PARAM_LIST[n_idx], max_iter=10000)
        
        # training & testing
        Logistic_model.fit(x_train, y_train)
        print(f"training acc = {round(Logistic_model.score(x_train, y_train), 5)}, testing acc = {round(Logistic_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = Logistic_model.score(x_test, y_test)
    print(f"Best prediction model (C = {REGULARIZE_PARAM_LIST[np.argmax(testing_result)]}) Testing Acc. = {round(testing_result.max(), 7) * 100}%\n")



    # Linear Support Vector Machine for Classification
    print(">> Linear Support Vector Classification")
    testing_result = np.empty(len(REGULARIZE_PARAM_LIST))
    for n_idx in range(len(REGULARIZE_PARAM_LIST)):
        print(f"For C = {REGULARIZE_PARAM_LIST[n_idx]}:  ", end='')
        
        # build model
        SVC_model = LinearSVC(C=REGULARIZE_PARAM_LIST[n_idx], max_iter=10000, dual=False)
        
        # training & testing
        SVC_model.fit(x_train, y_train)
        print(f"training acc = {round(SVC_model.score(x_train, y_train), 5)}, testing acc = {round(SVC_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = SVC_model.score(x_test, y_test)
    print(f"Best prediction model (C = {REGULARIZE_PARAM_LIST[np.argmax(testing_result)]}) Testing Acc. = {round(testing_result.max(), 7) * 100}%\n")


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
    print(f"Best prediction model (n_estimators = {NUM_OF_ESTIMATOR_LIST[np.argmax(testing_result)]}) Testing Acc. = {round(testing_result.max(), 7) * 100}%\n")
