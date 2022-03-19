import gaussian_estimators as ge
import numpy as np




if __name__ == '__main__':

#test ge.fit()
    X = np.arange(5)
    ub = ge.UnivariateGaussian()
    ub.fit(X)
    print(ub.mu_)
    print(ub.var_)
    print(ub.fitted_)

    b = ge.UnivariateGaussian(True)
    b.fit(X)
    print(b.mu_)
    print(b.var_)
    print(b.fitted_)

#test ge.pdf
    Y = np.arange(9)
    print(ub.pdf(Y))
    print(b.pdf(Y))

#test loglikelihood
    print(ge.UnivariateGaussian.log_likelihood(2,2,np.arange(5)))
