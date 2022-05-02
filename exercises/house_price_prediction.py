from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import IMLearn.utils.utils as ut
import IMLearn.learners.regressors.linear_regression as lg

PRESENT_YEAR = 2022
NUM_OF_ZIP_AREAS = 10
DEFAULT_ZEROS = 0000000000
TRAINING_PRECENT = 0.75


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    all_data = pd.read_csv (filename)

    #check invald data
    all_data = all_data.dropna(how='any')
    all_data = all_data.loc[(all_data != 0).any(axis=1)]

    features = ('bedrooms', 'bathrooms', 'sqft_lot', 'sqft_lot15') #decide what to delete
    for feature in features:
        all_data = all_data.drop(all_data[all_data[feature] <= 0].index)

    #update data to fit model
    prices = all_data['price']
    house_features = all_data
    house_features['yr_sold'] = house_features.date.str.slice(0,4).astype(str).astype(int)
    house_features['month_sold'] = house_features.date.str.slice(4,6).astype(str).astype(int)
    house_features['day_sold'] = house_features.date.str.slice(6,8).astype(str).astype(int)
    house_features['precent_above'] = house_features['sqft_above'] / house_features['sqft_living']
    house_features['precent_basement'] = house_features['sqft_basement'] / house_features['sqft_living']

    #divide price for zipcode area
    house_features['zip_area'] = house_features.zipcode // NUM_OF_ZIP_AREAS
    house_features = pd.get_dummies(house_features, columns=['zip_area'])

    #if the house wasn't wenovated, we check for when it was built
    yr_renovated = house_features[['yr_built', 'yr_renovated']].max(axis=1)
    house_features['yrs_since_last_renovation'] = PRESENT_YEAR - yr_renovated

    #drop unnecessary features
    house_features = house_features.drop(['id', 'price', 'date', 'sqft_above', 'sqft_basement',
                                          'yr_renovated','zipcode'], axis=1)
    return house_features, prices


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X_sd = X.std()
    y_sd = y.std()
    X_y_cov = X_sd*y_sd


    for i in range(X.shape[1]):
        feature = X.columns[i]
        filename = output_path + "/" + feature + ".jpg"
        X_y_cov[i] = X.iloc[:, i].cov(y)/X_y_cov[i]
        fig = go.Figure(
            go.Scatter(x=X.iloc[:, i], y=y, mode='markers', marker=dict(color="black"), showlegend=False),
            layout=go.Layout(title=f"r={X_y_cov[i]}",
                             xaxis_title= feature,
                             yaxis_title="price"))

        fig.write_image(filename)



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    house_features, prices = load_data('C:/Users/avner_suke883/IML.HUJI/datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(house_features, prices, "C:/Users/avner_suke883/IML.HUJI/avner/ex_2/Q2")

    # Question 3 - Split samples into training- and testing sets.
    hf_train, prices_train, hf_test, prices_test = ut.split_train_test(house_features, prices, TRAINING_PRECENT)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    #create linear regression model
    lg_houses = lg.LinearRegression()

    precents = np.arange(0.1, 1, 0.01)
    losses_mean = np.zeros(precents.shape[0])
    losses_std = np.zeros(precents.shape[0])

    for i in range(precents.shape[0]):

        losses = np.zeros(10)
        p = precents[i]
        for j in np.arange(10):
            hf_train_p = hf_train.sample(frac=p)
            prices_train_p = prices_train[hf_train_p.index]
            lg_houses._fit(hf_train_p.to_numpy(), prices_train_p.to_numpy())

            losses[j] = lg_houses._loss(hf_test.to_numpy(), prices_test.to_numpy())
        losses_mean[i] = losses.mean()
        losses_std[i] = losses.std()

    print(losses_mean)
    print(losses_std)

    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    data = go.Scatter(x=precents, y=losses_mean, mode="markers+lines", name="Mean Losses", line=dict(dash="dash"),
                      marker=dict(color="green", opacity=.7))

    # compute error ribbon:
    down_ribbon = go.Scatter(x=precents, y=losses_mean - 2 * (losses_std), fill=None, mode="lines",
                             line=dict(color="lightgrey"), showlegend=False)
    up_ribbon = go.Scatter(x=precents, y=losses_mean + 2 * (losses_std), fill='tonexty', mode="lines",
                           line=dict(color="lightgrey"), showlegend=False)

    fig = go.Figure(data=(data, down_ribbon, up_ribbon)) \
        .update_layout(title_text="Change in loss over different sampling precentage") \
        .update_xaxes(title_text="sampling precentage P") \
        .update_yaxes(title_text="loss for model trained on P precent of the data")

    fig.show()
