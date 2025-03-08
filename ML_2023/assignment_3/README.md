# Homework 3
Here are my solutions to the programming problems in Hw3. <br>

1. **<i>Problem 1</i>**：
    - Suppose we have the following dataset：

        | $x=(x_1, x_2)$ | $(1, 1)$ | $(1, 3)$ | $(5, 2)$ | $(7, 4)$ | $(9, 2)$ |
        |     :----:     |  :----:  |  :----:  |  :----:  |  :----:  |  :----:  |
        |      $y$       |   $2$    |   $5$    |   $3$    |   $4$    |   $8$    |

    - use Ridge to determine the best predicted model when
        <br>$\hat{y}(x, \mathbf{w}) = w_0 + w_1x_1 + w_2x_2$
    - use Lasso to determine the best predicted model when
        <br>$\hat{y}(x, \mathbf{w}) = w_0 + w_1x_1 + w_2x_2$
    - Write your own program and don’t use the toolboxes of Ridge and Lasso.
    <br><br>

    **<i>My Solution's Result</i> (outputs of "problem_01.py")**：
    <br>
    ```
    >> Least Square Linear Regression :
    Optimal weights = [2.42763158 0.40131579 0.05263158]

    >> Ridge Regression :
    Optimal weights (lambda=0) = [2.42763158 0.40131579 0.05263158]
    Optimal weights (lambda=0.01) = [2.42763158 0.40131579 0.05263158]
    Optimal weights (lambda=0.1) = [2.42763158 0.40131579 0.05263158]
    Optimal weights (lambda=1) = [2.46839654 0.39387891 0.0499002 ]
    Optimal weights (lambda=10) = [2.75837743 0.33686067 0.03835979]

    >> Lasso Regression :
    Optimal weights (lambda=0) = [2.42763158 0.40131579 0.05263158]
    Optimal weights (lambda=0.01) = [2.43228618 0.40129934 0.05072368]
    Optimal weights (lambda=0.1) = [2.47417763 0.40115132 0.03355263]
    Optimal weights (lambda=1) = [2.62109375 0.38671875 0.        ]
    Optimal weights (lambda=10) = [3.4296875 0.2109375 0.       ]
    ```
    ( I write two functions to implement Ridge Regression, but I only used one. )<br><br>
