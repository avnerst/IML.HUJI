from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SPLIT_RATIO = (2 / 3)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # create and split data
    # X = np.random.uniform(-1.2, 2, n_samples) # todo: delete
    X = np.linspace(-1.2, 2, num=n_samples)
    y_noiseless = ((X+3)*(X+2)*(X+1)*(X-1)*(X-2))
    eps = np.random.normal(0, noise, n_samples)
    y = y_noiseless + eps
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), SPLIT_RATIO)

    X_train, y_train, X_test, y_test = np.asarray(X_train).flatten(), np.asarray(y_train).flatten()\
        , np.asarray(X_test).flatten(), np.asarray(y_test).flatten()

    # scatter
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=y_noiseless,
                             mode='markers',
                             name='y noiseless'))
    fig.add_trace(go.Scatter(x=X_train, y=y_train,
                             mode='markers',
                             name='y train'))
    fig.add_trace(go.Scatter(x=X_test, y=y_test,
                             mode='markers',
                             name='y test'))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_losses = np.zeros(11)
    validation_losses = np.zeros(11)

    # iterate over different polynomial degrees
    for k in np.arange(11):
        pf_model = PolynomialFitting(k)
        train_losses[k], validation_losses[k] = cross_validate(pf_model, X_train, y_train, mean_square_error)

    # scatter
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(11), y=train_losses,
                             mode='lines + markers',
                             name='train losses'))
    fig.add_trace(go.Scatter(x=np.arange(11), y=validation_losses,
                             mode='lines + markers',
                             name='validation losses'))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(validation_losses)
    pf_model = PolynomialFitting(k)
    pf_model.fit(X_train, y_train)
    print(f"results for {noise} noise")
    print(f"the k which achieved the lowest validation error was: {k}")
    print(f"the test error for this model is: {np.round(pf_model._loss(X_test, y_test), 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    X_train = np.asarray(X)[:51]
    y_train = np.asarray(y)[:51]
    X_test = np.asarray(X)[51:]
    y_test = np.asarray(y)[51:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0, 4, num=n_evaluations)

    rr_train_losses = np.zeros(n_evaluations)
    rr_validation_losses = np.zeros(n_evaluations)
    lr_train_losses = np.zeros(n_evaluations)
    lr_validation_losses = np.zeros(n_evaluations)

    # iterate over different polynomial degrees
    for i in np.arange(n_evaluations):
        # create both models
        rr_model = RidgeRegression(lambdas[i])
        lr_model = Lasso(lambdas[i])
        rr_train_losses[i], rr_validation_losses[i] = cross_validate(rr_model, X_train, y_train, mean_square_error)
        lr_train_losses[i], lr_validation_losses[i] = cross_validate(lr_model, X_train, y_train,
                                                                     mean_square_error)

    # scatter
    fig = go.Figure()
    # add ridge losses
    fig.add_trace(go.Scatter(x=lambdas, y=rr_train_losses,
                             mode='lines + markers',
                             name='ridge train losses'))
    fig.add_trace(go.Scatter(x=lambdas, y=rr_validation_losses,
                             mode='lines + markers',
                             name='ridge validation losses'))
    # add lasso losses
    fig.add_trace(go.Scatter(x=lambdas, y=lr_train_losses,
                             mode='lines + markers',
                             name='lasso train losses'))
    fig.add_trace(go.Scatter(x=lambdas, y=lr_validation_losses,
                             mode='lines + markers',
                             name='lasso validation losses'))
    fig.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lambdas = np.zeros(2)
    best_lambdas[0] = lambdas[np.argmin(rr_validation_losses)]
    best_lambdas[1] = lambdas[np.argmin(lr_validation_losses)]

    print("the regularization parameter which achieved the lowest train error for "
          f"the ridge regression model was: {best_lambdas[0]}")
    print("the regularization parameter which achieved the lowest train error for "
          f"the lasso regression model was: {best_lambdas[1]}")

    for lam in best_lambdas:
        rr_model = RidgeRegression(lam=lam)
        lr_model = Lasso(alpha=lam)
        ls_model = LinearRegression()

        # fit models
        rr_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        ls_model.fit(X_train, y_train)

        # calc test errors
        rr_error = rr_model.loss(X_test, y_test)
        lr_error = mean_square_error(lr_model.predict(X_test), y_test)
        ls_error = ls_model.loss(X_test, y_test)

        print(f"test error for regulariztion parameter {lam} for ridge resgression model is {rr_error}")
        print(f"test error for regulariztion parameter {lam} for lasso resgression model is {lr_error}")
        print(f"test error for regulariztion parameter {lam} for linear resgression model is {ls_error}")




if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()