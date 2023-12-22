# Linear Regression, Ridge Regression, Lasso Regression + Diabetes Dataset
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


ALPHA_PARAM_LIST = [0.01, 0.1, 1, 10, 100]


if __name__ == "__main__":
    # data reading
    digits_data = load_diabetes()

    # splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(digits_data.data, digits_data.target, test_size=0.25, random_state=10)


    # Linear Regression
    print(">> Linear Regression")
    
    # build model
    LR_model = LinearRegression()
    
    # training & testing
    LR_model.fit(x_train, y_train)
    print(f"training acc = {round(LR_model.score(x_train, y_train), 5)}, testing acc = {round(LR_model.score(x_test, y_test), 5)}")
    print(f"Testing Acc. = {round(LR_model.score(x_test, y_test), 5) * 100}%\n")


    # Ridge Regression
    print(">> Ridge Regression")
    testing_result = np.empty(len(ALPHA_PARAM_LIST))
    for n_idx in range(len(ALPHA_PARAM_LIST)):
        print(f"For alpha = {ALPHA_PARAM_LIST[n_idx]}:  ", end='')
        
        # build model
        Ridge_model = Ridge(alpha=ALPHA_PARAM_LIST[n_idx], max_iter=10000)
        
        # training & testing
        Ridge_model.fit(x_train, y_train)
        print(f"training acc = {round(Ridge_model.score(x_train, y_train), 5)}, testing acc = {round(Ridge_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = Ridge_model.score(x_test, y_test)
    print(f"Best prediction model (alpha = {ALPHA_PARAM_LIST[np.argmax(testing_result)]}) Testing Acc. = {round(testing_result.max(), 7) * 100}%\n")


    # Lasso Regression
    print(">> Lasso Regression")
    testing_result = np.empty(len(ALPHA_PARAM_LIST))
    for n_idx in range(len(ALPHA_PARAM_LIST)):
        print(f"For alpha = {ALPHA_PARAM_LIST[n_idx]}:  ", end='')
        
        # build model
        Lasso_model = Lasso(alpha=ALPHA_PARAM_LIST[n_idx], max_iter=10000)
        
        # training & testing
        Lasso_model.fit(x_train, y_train)
        print(f"training acc = {round(Lasso_model.score(x_train, y_train), 5)}, testing acc = {round(Lasso_model.score(x_test, y_test), 5)}")
        
        # save result
        testing_result[n_idx] = Lasso_model.score(x_test, y_test)
    print(f"Best prediction model (alpha = {ALPHA_PARAM_LIST[np.argmax(testing_result)]}) Testing Acc. = {round(testing_result.max(), 7) * 100}%\n")
