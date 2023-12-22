# Logistic, LinearSVC, GaussianNB vs KNN + Digits Dataset
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


NUM_OF_NEIGHBORS_LIST = range(2, 11)
REGULARIZE_PARAM_LIST = [0.001, 0.01, 0.1, 1, 10, 100]


if __name__ == "__main__":
    # data reading
    digits_data = load_digits()

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(digits_data.data, digits_data.target, test_size=0.25, random_state=10)


    # KNN
    print(">> KNN")
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
    print(f"Best prediction model (n_neighbors = {NUM_OF_NEIGHBORS_LIST[np.argmax(testing_result)]}) Testing Acc. = {round(testing_result.max(), 7) * 100}%\n")


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


    # Guassian Naive Bayes
    print(">> Guassian Naive Bayes")
    
    # build model
    NB_model = GaussianNB()
    
    # training & testing
    NB_model.fit(x_train, y_train)
    print(f"training acc = {round(NB_model.score(x_train, y_train), 5)}, testing acc = {round(NB_model.score(x_test, y_test), 5)}")
    print(f"Testing Acc. = {round(NB_model.score(x_test, y_test), 7) * 100}%\n")
