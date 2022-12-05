import os
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal


def log_normal(x, mu=0, sigma=1):

    mu = np.ones_like(x) * mu
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)

    return np.sum(np.log(numerator/denominator))


def generate_traceplots(samples, prefix=''):
    """Create and save traceplots for each parameter from sampling algorithm"""
    n_samples = len(samples)-1
    n_params = len(samples[0])
    param_names = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(8, 11))
    plt.tight_layout()

    for j in range(n_params):
        marginal_samples = [samples[i][j] for i in range(len(samples))]
        ax[j].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples)
        ax[j].set_ylabel(param_names[j])
    plt.xlabel('number of samples')
    outfile = prefix + 'sampled_traceplot.png'
    plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    plt.show(); plt.close()

    # for j in range(n_params):
    #     marginal_samples = [samples[i][j] for i in range(len(samples))]
    #     plt.scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples)
    #     outfile = prefix + 'sampled_traceplot_' + param_names[j] + '.png'
    #     plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    #     plt.show(); plt.close()


def generate_posterior_histograms(samples, prefix=''):
    """Create and save marginal histograms for each parameter from sampled posterior"""
    n_params = len(samples[0])
    param_names = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    fig, ax = plt.subplots(6, 1, figsize=(8, 12))
    plt.tight_layout()

    for j in range(n_params):
        marginal_samples = [samples[i][j] for i in range(len(samples))]
        ax[j].hist(marginal_samples, bins=50)
        ax[j].set_xlabel(param_names[j])
    outfile = prefix + 'sampled_histogram.png'
    plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    plt.show(); plt.close()

    # for j in range(n_params):
    #     marginal_samples = [samples[i][j] for i in range(len(samples))]
    #     plt.hist(marginal_samples, bins=50)
    #     outfile = prefix + 'sampled_histogram_' + param_names[j] + '.png'
    #     plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    #     plt.show(); plt.close()


def log_normal(x, mu=0, sigma=1):
    """Helper function to query pdf from a log std normal"""
    mu = np.ones_like(x) * mu
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)

    return np.sum(np.log(numerator/denominator))


def negative_log_prob(data, theta):
    """Compute negative log probability evaluated at a given theta"""
    n, k = data[['X1', 'X2']].shape
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    mu = np.array([mu1, mu2])
    gam = np.array([gam1, gam2])

    # center data
    centered_data = center_data(mu, gam, tau, data)

    # compute log likelihood
    norm_term = (n*k*0.5) * np.log(2*math.pi)
    sigma_term = (n + 1) * np.log(sigma_sq)
    posterior_term = np.sum(np.power(np.linalg.norm(centered_data, ord=2, axis=1), 2))
    output = norm_term + sigma_term + (0.5/sigma_sq)*posterior_term

    return output


def joint_posterior_density(data, theta):
    """Compute posterior density given params theta and data"""
    pi_list = []
    n_groups = len(data['group'].unique())

    for i in range(1, n_groups+1):
        mu, sigma = fetch_param_for_group(theta, group_id=i)
        x_grp = fetch_data_for_group(data, group_id=i)
        n_data_points = x_grp.shape[0]

        for j in range(n_data_points):
            x = np.array(x_grp[j:j+1])
            dist = multivariate_normal(mean=mu, cov=sigma)
            pi_list.append(dist.pdf(x))

    return np.prod(pi_list)*(1/theta[0])


def fetch_data_for_group(data, group_id):
    """Helper function to fetch gene expressions within specified group"""
    filtered_data = data[data['group'] == group_id].copy()
    return filtered_data[['X1', 'X2']]


def fetch_param_for_group(theta, group_id):
    """Helper function to fetch parameters for specified gene group"""
    if group_id == 1:
        mean = np.array([theta[2], theta[3]])

    if group_id == 2:
        mean = np.array([theta[4], theta[5]])

    if group_id == 3:
        mean = 0.5*np.array([theta[2], theta[3]]) + 0.5*np.array([theta[4], theta[5]])

    if group_id == 4:
        mean = theta[1]*np.array([theta[2], theta[3]]) + (1-theta[1])*np.array([theta[4], theta[5]])

    return mean, theta[0]*np.eye(2)


def fetch_data(data):
    """Helper function to fetch data and group vector"""
    X = np.array(data[['X1', 'X2']])
    t = np.array(data[['group']]).flatten()
    return X, t


def center_data(mu, gam, tau, data):
    """Apply hierarchical logic to center data given theta"""

    means_a = np.repeat(mu, 4, axis=0)
    means_b = np.repeat(gam, 4, axis=0)
    means_c = 0.5 * np.repeat(mu, 8, axis=0) + 0.5 * np.repeat(gam, 8, axis=0)
    means_d = tau * np.repeat(mu, 8, axis=0) + (1 - tau) * np.repeat(gam, 8, axis=0)
    means = np.vstack((means_a, means_b, means_c, means_d))
    centered_data = data[['X1', 'X2']] - means

    return centered_data


def beta_reparameterization(mu, sigma_sq):
    """Reparameterize beta in terms of mean and variance"""
    alpha = (((1-mu)/sigma_sq) - (1/mu))*(mu**2)
    beta = alpha*((1/mu) - 1)
    return alpha, beta
