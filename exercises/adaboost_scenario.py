import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def my_decision_surface(predict, t, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], t)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                          marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip",
                          showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False,
                      opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


custom = [[0.0, "rgb(165,0,38)"],
          [0.1111111111111111, "rgb(215,48,39)"],
          [0.2222222222222222, "rgb(244,109,67)"],
          [0.3333333333333333, "rgb(253,174,97)"],
          [0.4444444444444444, "rgb(254,224,144)"],
          [0.5555555555555556, "rgb(224,243,248)"],
          [0.6666666666666666, "rgb(171,217,233)"],
          [0.7777777777777778, "rgb(116,173,209)"],
          [0.8888888888888888, "rgb(69,117,180)"],
          [1.0, "rgb(49,54,149)"]]

class_symbols = np.array(["circle", "x", "diamond"])
class_colors = lambda n: [custom[i] for i in np.linspace(0, len(custom) - 1, n).astype(int)]


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ab_model = AdaBoost(wl=DecisionStump, iterations=n_learners)
    ab_model._fit(train_X, train_y)
    train_loss = np.zeros(n_learners)
    test_loss = np.zeros(n_learners)
    for i in np.arange(n_learners) + 1:
        train_loss[i - 1] = ab_model.partial_loss(train_X, train_y, i)
        test_loss[i - 1] = ab_model.partial_loss(test_X, test_y, i)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(n_learners) + 1, y=train_loss, mode='markers + lines', name="train loss"))
    fig.add_trace(go.Scatter(x=np.arange(n_learners) + 1, y=test_loss, mode='markers + lines', name="test loss"))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    # T = [5, 50]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["cross", "diamond"])

    # create y array with 0 instead of -1
    m = test_y.shape[0]
    y_for_plot = np.zeros(m, dtype=int)
    y_for_plot[test_y == 1] = 1

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} learners" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([my_decision_surface(ab_model.partial_predict, t, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, symbol=symbols[y_for_plot],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)


    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries For Different Sized Ensembles Over Data With {noise} noise}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()

    # Question 3: Decision surface of best performing ensemble
    losses = np.zeros(n_learners)
    for t in np.arange(n_learners) + 1:
        losses[t-1] = ab_model.partial_loss(test_X, test_y, t)

    # find the best ensemble number
    t_learners = np.argmin(losses)
    t_learners += 1
    acc = accuracy(test_y, ab_model.partial_predict(test_X, t_learners))

    fig = go.Figure()
    fig.add_traces([my_decision_surface(ab_model.partial_predict, t_learners, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y, symbol=symbols[y_for_plot],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])

    fig.update_layout(title=f"Ensemble Of {t_learners} Learners, With Accuracy: {acc}", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()

    # Question 4: Decision surface with weighted samples
    D = (ab_model.D_ / np.max(ab_model.D_)) * 5

    # create y array with 0 instead of -1
    m = train_y.shape[0]
    y_for_plot = np.zeros(m, dtype=int)
    y_for_plot[train_y == 1] = 1

    fig = go.Figure()
    fig.add_traces([my_decision_surface(ab_model.partial_predict, ab_model.iterations_, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y, symbol=symbols[y_for_plot],
                                           colorscale=[custom[0], custom[-1]],
                                           size=D,
                                           line=dict(color="black", width=1)))])

    fig.update_layout(title=f" Full Ensemble, Showing Weighted Labels with {noise} noise", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
