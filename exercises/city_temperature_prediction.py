import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import IMLearn.utils.utils as ut
import IMLearn.learners.regressors.polynomial_fitting as pf

TRAINING_PRECENT = 0.75


def load_data(filename: str) -> (pd.DataFrame,pd.DataFrame):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    #load data and check for invalid arguments
    all_data = pd.read_csv(filename, parse_dates=['Date'])
    all_data['DayOfYear'] = all_data['Date'].dt.dayofyear
    all_data = all_data.drop(all_data[all_data['Temp'] < -20].index)

    #split to features and temp
    temp = all_data['Temp']
    features = all_data

    #drop unnecessary features
    all_data = all_data.drop(['Date', 'Day', 'Temp'], axis=1)

    return features, temp


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X, y = load_data('C:/Users/avner_suke883/IML.HUJI/datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    X['Temp'] = y
    X_israel = X[X['Country'] == 'Israel']

    #plot temprature over day of year, with year as discrete variable
    X_israel['Year'] = X_israel['Year'].astype(str)

    fig = px.scatter(X_israel, x="DayOfYear", y="Temp",
                    color='Year',
                     title="mean temprature for day of year"
                    )

    fig.show()

    #plot standard deviation of temprature in each month
    std_months = X_israel.groupby('Month').Temp.agg('std')

    fig = px.bar(std_months)
    fig.update_layout(
        yaxis_title="Month Standard Deviation"
    )
    fig.show()

    # Question 3 - Exploring differences between countries
    X_country_month = X.groupby(['Country', 'Month'], as_index=False).agg({'Temp': ['mean', 'std']})
    X_country_month.columns = ['Country', 'Month', 'mean', 'std']

    #plot
    fig = px.line(X_country_month, x='Month', y= 'mean',
                  color='Country',
                  error_y = 'std',
                  title= "mean temperature by month in different countries")
    fig.update_layout(yaxis_title = "mean temperature")
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    day_of_year_israel = X_israel['DayOfYear']
    temp_israel = X_israel['Temp']
    X_train, y_train, X_test, y_test = ut.split_train_test(day_of_year_israel, temp_israel, TRAINING_PRECENT)

    losses = np.zeros(10)
    for i in np.arange(10):
        k = i+1
        temp_poly_fit = pf.PolynomialFitting(k)
        temp_poly_fit._fit(X_train.to_numpy(), y_train.to_numpy())
        losses[i] = round(temp_poly_fit._loss(X_test.to_numpy(), y_test.to_numpy()), 2)
        print(f"test error for degree {k} is: {losses[i]}")

    fig = px.bar(x=(np.arange(10) + 1), y=losses)
    fig.update_layout(
        xaxis_title="k degree of fit",
        yaxis_title="loss",
        title="loss over increasing k degree of polynomial fit"
    )
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    k = np.argmin(losses)
    temp_poly_fit = pf.PolynomialFitting(k)
    temp_poly_fit._fit(day_of_year_israel.to_numpy(), temp_israel.to_numpy())
    X_jordan = X[X['Country'] == 'Jordan']
    X_south_africa = X[X['Country'] == 'South Africa']
    X_the_netherlands = X[X['Country'] == 'The Netherlands']

    #get temperatures
    y_jordan = X_jordan['Temp']
    y_south_africa = X_south_africa['Temp']
    y_the_netherlands = X_the_netherlands['Temp']

    #use only DayOfYear to fit polynomial model
    X_jordan = X_jordan['DayOfYear']
    X_south_africa = X_south_africa['DayOfYear']
    X_the_netherlands = X_the_netherlands['DayOfYear']

    #calculate loss
    loss = np.zeros(3)
    loss[0] = temp_poly_fit._loss(X_jordan.to_numpy(), y_jordan.to_numpy())
    loss[1] = temp_poly_fit._loss(X_south_africa.to_numpy(), y_south_africa.to_numpy())
    loss[2] = temp_poly_fit._loss(X_the_netherlands.to_numpy(), y_the_netherlands.to_numpy())
    country_loss = pd.DataFrame()
    country_loss["Country"] = ['Jordan', 'South Africa', 'The Netherlands']
    country_loss["Loss"] = loss

    #plot loss over countries
    fig = px.bar(country_loss, x='Country', y='Loss')
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Loss",
        title="loss for different countries of model trained on Israel weather"
    )
    fig.show()



