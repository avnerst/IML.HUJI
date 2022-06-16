from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # split train data into subgroups
    X_subgroups = np.array_split(X, cv)
    y_subgroups = np.array_split(y, cv)


    # iterate over different groups of k fold
    train_loss_sum = 0
    validation_loss_sum = 0
    for i in np.arange(cv):
        # create k-fold subgroups
        X_train_sub = np.concatenate(X_subgroups[:i] + X_subgroups[i + 1:])
        y_train_sub = np.concatenate(y_subgroups[:i] + y_subgroups[i + 1:])
        estimator.fit(X_train_sub, y_train_sub)

        # calc 2 losses
        train_loss_sum += scoring(y_train_sub, estimator.predict(X_train_sub))
        validation_loss_sum += scoring(y_subgroups[i],estimator.predict(X_subgroups[i]))

    # calc average losses
    train_score = train_loss_sum / cv
    validation_score = validation_loss_sum / cv

    return train_score, validation_score
