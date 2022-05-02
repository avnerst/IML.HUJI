from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        # fit classes vector
        self.classes_, class_index, class_inverse = np.unique(y, return_index=True, return_inverse=True)

        m = X.shape[0]
        d = X.shape[1]
        c = self.classes_.shape[0]

        # fit mu vector, fit pi vector
        self.mu_ = np.empty([c, d])
        self.pi_ = np.empty(c)

        for i in np.arange(c):
            k = self.classes_[i]
            n_k = sum(y == k)
            self.pi_[i] = (n_k / m)
            # get summed values of features for samples from class k
            X_k_sum = np.sum(X[y == k], axis=0)
            mu_k = X_k_sum / n_k
            self.mu_[i] = (mu_k)

        # fit cov
        X_mu = self.mu_[class_inverse]
        X_cov = (X - X_mu).T @ (X - X_mu)
        self.cov_ = X_cov / (m - c)

        # fit cov_inv
        self._cov_inv = np.linalg.inv(self.cov_)

        # mark model as fitted
        self.fitted_ = True


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        return np.take(self.classes_, indices=(np.argmax(self.likelihood(X), axis=1)))

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m = X.shape[0]
        d = X.shape[1]
        c = self.classes_.shape[0]
        cov_det = np.linalg.det(self.cov_)

        z = (1 / (np.sqrt(np.power((2 * np.pi), d) * cov_det)))
        X_pdf = np.empty([m, c])

        for i in np.arange(c):
            X_pdf[:, i] = z * np.exp( -0.5 * (np.diag((X - self.mu_[i]) @ self._cov_inv @ (X - self.mu_[i]).T)))

        return X_pdf * self.pi_


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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
