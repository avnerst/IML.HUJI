import mkl_random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

MEAN = 10
VAR = 1
SAMPLE_SIZE = 1000
INTERVAL = 10

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    x = np.random.normal(MEAN, VAR, SAMPLE_SIZE)
    ug = UnivariateGaussian()
    ug.fit(x)
    print(f"({ug.mu_}, {ug.var_}).")


    # Question 2 - Empirically showing sample mean is consistent
    samples = np.zeros((2, int(SAMPLE_SIZE/INTERVAL)))
    samples[0] = np.arange(10,SAMPLE_SIZE + 1, 10)

    #fir model for different sample sizes
    for i in range(samples[0].size):
        y = x[0:int(samples[0, i])]
        ug.fit(y)
        samples[1, i] = abs(MEAN - ug.mu_)

    #plot data
    fig = go.Figure(go.Scatter(x=samples[0], y=samples[1], mode='markers+lines', marker=dict(color="black"), showlegend=False),
         layout=go.Layout(title=r"$\text{(6) Estimation of Expectation Error As Function Of Number Of Samples}$",
                          xaxis_title=" number of samples",
                          yaxis_title="estimation of expectation error"))
    fig.show() #TODO: save graph as file



    # Question 3 - Plotting Empirical PDF of fitted model
    x_pdf = ug.pdf(x)
    fig = go.Figure(
        go.Scatter(x=x, y=x_pdf, mode='markers', marker=dict(color="black"), showlegend=False),
        layout=go.Layout(title=r"$\text{(6) PDF Of Samples According To Fitted Model}$",
                         xaxis_title="x",
                         yaxis_title="PDF(x)"))
    fig.show()  # TODO: save graph as file


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = mkl_random.multivariate_normal(mu, cov, SAMPLE_SIZE)
    multy = MultivariateGaussian()
    multy.fit(X)
    print("estimated mu:", multy.mu_)
    print("estimated cov:", multy.cov_)

    #Question 5 - Likelihood evaluation
    vals_for_mu = np.linspace(-10, 10, 200)
    loglike

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
