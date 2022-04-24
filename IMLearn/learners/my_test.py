from scipy.stats import multivariate_normal

import gaussian_estimators as ge
import numpy as np
import IMLearn.metrics.loss_functions as lf
import IMLearn.learners.regressors.linear_regression as lr
import exercises.house_price_prediction as hpp
import pandas as pd
import IMLearn.learners.regressors.polynomial_fitting as pf
import exercises.city_temperature_prediction as ctp


def test_MLE():
    y_true = np.arange(10)
    y_pred = np.array([3,5,1,22,90,88,-15,-2,0,444])
    print(lf.mean_square_error(y_true,y_pred))

def test_lr_fit():
    linear_with_intercept = lr.LinearRegression(True)
    linear_without_intercept = lr.LinearRegression(False)
    X = np.array([[2,3,4],[2,5,7],[0,0,3],[1,1,1]])
    y = np.arange(4)
    linear_with_intercept.fit(X,y)
    linear_without_intercept.fit(X,y)
    print(f'with intercept {linear_with_intercept.coefs_}')
    print(f'without intercept {linear_without_intercept.coefs_}')
    # print(f'loss with intercept {linear_with_intercept._loss(X,y)}')
    # print(f'loss without intercept {linear_without_intercept._loss(X, y)}')

def test_hpp():
    feats, prices = hpp.load_data('C:/Users/avner_suke883/IML.HUJI/datasets/house_prices.csv')
    # print(feats)
    # print(prices)
    # print(feats[['long']])
    print(feats.columns)
    # hpp.feature_evaluation(feats,prices, "C:/Users/avner_suke883/IML.HUJI/avner/ex_2/Q2")
    # print(feats.iloc[:, 3].cov(prices))

def test_poly_fit():
    X = np.array([1,2,3])
    y = np.array([3,7,13])
    poly_model = pf.PolynomialFitting(2)
    poly_model._fit(X,y)
    print(poly_model.coefs_)
    # X, y = ctp.load_data('C:/Users/avner_suke883/IML.HUJI/datasets/City_Temperature.csv')

def quiz_2():
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print(lf.mean_square_error(y_true, y_pred))

def test_miss_calssification():
    x1 = np.array([1,2,1])
    x2 = np.array([2,1,4])
    print(f" regular:{lf.misclassification_error(x1,x2,False)}")
    print(f" normalized:{lf.misclassification_error(x1,x2)}")

if __name__ == '__main__':
    test_miss_calssification()






