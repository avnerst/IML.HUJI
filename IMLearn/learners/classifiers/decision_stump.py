from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error



class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # initialize variables
        d = X.shape[1]
        thrs = np.zeros(d)
        losses = np.zeros(d)
        signs = np.ones(d)

        for i in np.arange(d):
            # calculate for sign = 1
            thrs[i], losses[i] = self._find_threshold(X[:, i], y, 1)
            # calculate for sign = -1
            negative_thr, negative_loss = self._find_threshold(X[:, i], y, -1)
            # check which sign preformed better
            if negative_loss < losses[i]:
                thrs[i], losses[i], signs[i] = negative_thr, negative_loss, -1

        self.j_ = np.argmin(losses)
        self.threshold_ = thrs[self.j_]
        self.sign_ = signs[self.j_]
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # initialize predictions
        m = X.shape[0]
        y_hat = np.full(m, -1 * self.sign_)

        # update predictions
        y_hat[X[:, self.j_] >= self.threshold_] = self.sign_

        return y_hat



    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        m = values.shape[0]
        all_sign = np.full(m, sign)

        # initialize loss
        min_loss = self._weighted_loss(labels, all_sign)
        thr = -np.inf

        unique = np.unique(values)

        for val in unique:
            y_hat = np.full(m, sign)
            y_hat[values < val] = -sign
            loss = self._weighted_loss(labels, y_hat)
            if loss < min_loss:
                thr = val
                min_loss = loss

        # check if loss is simillar when they are all -sign
        all_minus_sign = all_sign = np.full(m, -sign)
        all_minus_loss = self._weighted_loss(labels, all_minus_sign)
        if all_minus_loss == min_loss:
            thr = np.inf

        return thr, min_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))

    def _weighted_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        returns weighted loss of y_hat predictions over y real labels
        """
        m = y.shape[0]
        indices = (y * y_hat < 0)
        return np.sum(np.abs(y[indices])) / m
