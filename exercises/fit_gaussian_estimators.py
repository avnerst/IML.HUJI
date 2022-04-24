import mkl_random
import numpy

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

MEAN = 10
VAR = 1
SAMPLE_SIZE = 1000
INTERVAL = 10
GRID_SIZE = 200

def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    x = np.random.normal(MEAN, VAR, SAMPLE_SIZE)
    ug = UnivariateGaussian()
    ug.fit(x)
    print(f"({ug.mu_}, {ug.var_}).")


    # Question 2 - Empirically showing sample mean is consistent
    samples = np.zeros((2, int(SAMPLE_SIZE/INTERVAL)))
    samples[0] = np.arange(10,SAMPLE_SIZE + 1, 10)

    #fit model for different sample sizes
    for i in range(samples[0].size):
        y = x[0:int(samples[0, i])]
        ug.fit(y)
        samples[1, i] = abs(MEAN - ug.mu_)

    #plot data
    fig = go.Figure(go.Scatter(x=samples[0], y=samples[1], mode='markers+lines', marker=dict(color="black"), showlegend=False),
         layout=go.Layout(title=r"$\text{(6) Estimation of Expectation Error As Function Of Number Of Samples}$",
                          xaxis_title=" number of samples",
                          yaxis_title="estimation of expectation error"))
    fig.show()



    # Question 3 - Plotting Empirical PDF of fitted model
    x_pdf = ug.pdf(x)
    fig = go.Figure(
        go.Scatter(x=x, y=x_pdf, mode='markers', marker=dict(color="black"), showlegend=False),
        layout=go.Layout(title=r"$\text{(6) PDF Of Samples According To Fitted Model}$",
                         xaxis_title="x",
                         yaxis_title="PDF(x)"))
    fig.show()

def calc_llh_for_pair(pair,cov, X):
    mu = np.array([pair[0], 0, pair[1], 0])
    return MultivariateGaussian.log_likelihood(mu, cov, X)


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, SAMPLE_SIZE)
    multy = MultivariateGaussian()
    multy.fit(X)
    print(multy.mu_)
    print(multy.cov_)

    #Question 5 - Likelihood evaluation
    vals_for_axis = np.linspace(-10, 10, GRID_SIZE)
    pairs_for_axis = np.array(np.meshgrid(vals_for_axis,vals_for_axis)).T.reshape(-1,2)
    log_likelihood_arr = np.apply_along_axis(calc_llh_for_pair,1 ,pairs_for_axis, cov, X)
    log_likelihood_matrix = np.reshape(log_likelihood_arr, (GRID_SIZE, GRID_SIZE))
    #plot data
    fig = go.Figure(go.Heatmap(x=vals_for_axis, y=vals_for_axis, z=log_likelihood_matrix,type= 'heatmap',  colorscale='Viridis'),
        layout=go.Layout(title=r"$\text{(6) log-likelihood calculated for different f1, f3 values}$",
                              xaxis_title="f3",
                              yaxis_title="f1"))
    fig.show()

    # Question 6 - Maximum likelihood
    grid_diff = (10 - (-10)) / (GRID_SIZE-1)
    argmax_coordinates = np.array(np.unravel_index(log_likelihood_matrix.argmax(), log_likelihood_matrix.shape))
    f1 = np.around(-10 + grid_diff * float(argmax_coordinates[0]), decimals=3)
    f3 = np.around(-10 + grid_diff * float(argmax_coordinates[1]), decimals=3)
    print(f1)
    print(f3)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
