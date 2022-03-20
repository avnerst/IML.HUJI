from scipy.stats import multivariate_normal

import gaussian_estimators as ge
import numpy as np


def test_uni():
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

def test_multi():
    print("test multivariate")
    x = np.array([[1, 2], [1, 1], [2, 15]])
    m = ge.MultivariateGaussian()
    m.fit(x)
    # print(x)
    # print(m.cov_)
    # print(m.mu_)
    t = np.array([[1,1], [2,2], [3,3]])
    # print(t)
    # print(m.pdf(t))
    print(ge.MultivariateGaussian.log_likelihood(m.mu_, m.cov_, t))
    for t_i in t:
        print(m.pdf(np.array([t_i])))

# ========== MultivariateGaussian ============

def test_multy_pdf(X):
    print("X : \n", X)
    print("fitted_: ", multy.fitted_)
    print("====== FIT ==========")
    multy.fit(X)
    print("mu_: \n", multy.mu_, multy.mu_.shape)
    print("cov_: \n", multy.cov_, multy.cov_.shape)
    print("fitted_: ", multy.fitted_)
    print("")
    print("====== PDF ==========")
    my_pdf = multy.pdf(X)
    scipy_pdf = multivariate_normal.pdf(X, mean=multy.mu_, cov=multy.cov_)
    print("multy.pdf: ", my_pdf, my_pdf.shape)
    print("scipy pdf: ", scipy_pdf)
    print("scipy result == my result: ", np.allclose(my_pdf,scipy_pdf))
    print("")

def test_multy_loglikelihood(mu, cov):
    print("====== loglikelihood ==========")
    X = np.array([[ 3.48421759e-01,  1.59648054e+01,  6.27508725e+00],
 [-1.18715285e+00,  1.49710265e+01,  8.72698139e+00],
 [-7.12795148e-01,  1.43093831e+01,  7.24666632e+00],
 [-7.41590178e-01,  1.64028792e+01,  6.83946392e+00],
 [-8.76695068e-01,  1.23500578e+01,  7.52108605e+00],
 [ 1.16338044e+00,  1.37171868e+01,  8.35907284e+00],
 [ 2.68765531e-01,  1.76922704e+01,  6.52737163e+00],
 [ 1.13105658e+00,  1.42448585e+01,  5.81726528e+00],
 [-9.30955465e-03,  1.37323256e+01,  7.21710167e+00],
 [ 4.34746241e-01,  1.57538403e+01,  6.29936432e+00],
 [-6.38122102e-01,  1.50497513e+01,  7.09916426e+00],
 [ 1.77014521e+00,  1.45433467e+01,  6.95241512e+00],
 [-1.16014637e+00,  1.62643037e+01,  7.14984288e+00],
 [-7.16407327e-01,  1.49817318e+01,  6.83694925e+00],
 [ 2.80810758e-01,  1.41827052e+01,  4.90927949e+00],
 [-1.21626390e+00,  1.47210685e+01,  9.53007019e+00],
 [ 3.56990107e-01,  1.40237485e+01,  5.37826224e+00],
 [-2.09433893e+00,  1.54307599e+01,  6.28051900e+00],
 [-7.98669075e-01,  1.39546683e+01,  7.97382524e+00],
 [-8.26245778e-01,  1.62366544e+01,  7.98084171e+00],
 [-1.70320655e+00,  1.53260502e+01,  7.46205007e+00],
 [-8.96041825e-01,  1.37247314e+01,  6.88039407e+00],
 [ 1.08828143e+00,  1.62193069e+01,  5.26865261e+00],
 [-1.25791821e+00,  1.22553550e+01,  7.30536337e+00],
 [-4.68928365e-02,  1.47343148e+01,  6.61731179e+00],
 [ 5.82121565e-01,  1.48588665e+01,  6.28805614e+00],
 [-1.98045939e-01,  1.62359043e+01,  6.68750316e+00],
 [ 1.69841794e+00,  1.54285643e+01,  8.47605357e+00],
 [ 8.32870239e-01,  1.55131916e+01,  7.34042625e+00],
 [ 5.69364871e-01, 1.49345735e+01,  8.10730543e+00],
 [-1.36750882e-01,  1.51732782e+01,  6.96058848e+00],
 [ 4.33415040e-01,  1.43644615e+01,  7.51710321e+00],
 [ 7.24268328e-01, 1.51061155e+01,  9.47591134e+00],
 [ 6.95382232e-01,  1.53819213e+01,  5.92094942e+00],
 [ 9.61005758e-01,  1.34096853e+01,  9.71883342e+00],
 [-9.52174696e-01,  1.52219680e+01,  7.02996983e+00],
 [-8.64858186e-01,  1.62335541e+01,  7.15175905e+00],
 [-2.92731883e+00,  1.53119814e+01,  6.86271827e+00],
 [-1.76630142e+00,  1.63863805e+01,  7.64436933e+00],
 [ 9.95192819e-01,  1.36842212e+01,  8.48954620e+00],
 [-1.01417304e-01,  1.42666817e+01, 6.81672829e+00],
 [ 1.49860908e+00,  1.42280413e+01,  8.61141685e+00],
 [-5.50222628e-01,  1.48407688e+01,  6.47157551e+00],
 [ 1.41397121e+00,  1.39250773e+01,  6.86769683e+00],
 [-5.29529813e-01,  1.32992588e+01,  6.55246165e+00],
 [-1.05487798e+00,  1.51689624e+01,  6.56448734e+00],
 [ 1.35159458e-01,  1.50495646e+01,  6.77839994e+00],
 [-1.19595389e+00,  1.53111355e+01,  5.74635805e+00],
 [-5.10547086e-01,  1.60293617e+01,  6.70165581e+00],
 [ 1.95548119e+00,  1.54295245e+01,  7.91048071e+00]])
    ll = ge.MultivariateGaussian.log_likelihood(mu, cov, X)
    print(ll, type(ll))


if __name__ == '__main__':
    #
    # # test_multi()
    # Y2 = np.random.multivariate_normal(np.array([0, 15, 7]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), size=50,
    #                                    check_valid='warn', tol=1e-8)
    # multy = ge.MultivariateGaussian()
    # test_multy_pdf(Y2)

    mu = np.array([0, 15, 7])
    cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    test_multy_loglikelihood(mu, cov)



