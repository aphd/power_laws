import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF


def normal_distribution(mu=0, variance=1, points=100):
    x_range = 5
    sigma = math.sqrt(variance)
    x = np.linspace(mu - x_range * sigma, mu + x_range * sigma, points)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()


def log_normal_distribution(alpha=1, points=100, log_scale=False):
    x = np.linspace(0, 100, points)
    plt.plot(x, stats.lognorm.pdf(x, alpha))
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.show()


def pareto_distribution(alpha=1, points=100, log_scale=False):
    x = np.linspace(0, 100, points)
    plt.plot(x, stats.pareto.pdf(x, alpha))
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    plt.show()


def log(x, a, b, h, k):
    return a * np.log10(b * x + h) + k


DATA_DIR = './data/blockchain'
if __name__ == '__main__':
    # normal_distribution(points=100)
    # log_normal_distribution()
    # pareto_distribution(log_scale=True)
    df = pd.read_csv('{}/{}'.format(DATA_DIR, 'monero-forum.csv'), delimiter=',')
    df = df[df['AvgCyclomaticStrict'].notnull()]
    x = np.array(df.AvgCyclomaticStrict.values)
    # Empirical data
    ecdf = ECDF(x)
    plt.plot(ecdf.x, 1 - ecdf.y, 'bo')
    # Pareto Distribution
    pareto_param = stats.pareto.fit(x)  # fit the sample data
    pdf_pareto_fitted = stats.pareto.pdf(x, pareto_param[0])  # fitted distribution
    ecdf = ECDF(pdf_pareto_fitted)
    plt.plot(ecdf.x, 1 - ecdf.y, 'rx')
    # # Log Normal Distribution
    lognormal_param = stats.lognorm.fit(x)  # fit the sample data
    pdf_log_normal_fitted = stats.lognorm.pdf(x, lognormal_param[0],
                                              loc=lognormal_param[1],
                                              scale=lognormal_param[2]
                                              )  # fitted distribution
    ecdf = ECDF(pdf_log_normal_fitted)
    plt.plot(ecdf.x, 1 - ecdf.y, 'gs')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend(['Empirical', 'Pareto', 'LogNormal'])
    plt.show()
