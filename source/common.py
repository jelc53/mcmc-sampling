import os
import numpy as np
import matplotlib.pyplot as plt


def generate_traceplots(samples, prefix=''):
    """Create and save traceplots for each parameter from sampling algorithm"""
    n_samples = len(samples)-1
    n_params = len(samples[0])
    param_names = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    for j in range(n_params):
        marginal_samples = [samples[i][j] for i in range(len(samples))]
        plt.scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples)
        outfile = prefix + 'sampled_traceplot_' + param_names[j] + '.png'
        plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
        plt.show(); plt.close()


def generate_posterior_histograms(samples, prefix=''):
    """Create and save marginal histograms for each parameter from sampled posterior"""
    n_params = len(samples[0])
    param_names = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    for j in range(n_params):
        marginal_samples = [samples[i][j] for i in range(len(samples))]
        plt.hist(marginal_samples, bins=50)
        outfile = prefix + 'sampled_histogram_' + param_names[j] + '.png'
        plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
        plt.show(); plt.close()


def fetch_data_for_group(data, group_id):
    """Helper function to fetch gene expressions within specified group"""
    filtered_data = data[data['group'] == group_id].copy()
    return filtered_data[['X1', 'X2']]


def fetch_param_for_group(theta, group_id):
    """Helper function to fetch parameters for specified gene group"""
    if group_id == 1:
        mu = np.array([theta[2], theta[3]])

    if group_id == 2:
        mu = np.array([theta[4], theta[5]])

    if group_id == 3:
        mu = 0.5*np.array([theta[2], theta[3]]) + 0.5*np.array([theta[4], theta[5]])

    if group_id == 4:
        mu = theta[1]*np.array([theta[2], theta[3]]) + (1-theta[1])*np.array([theta[4], theta[5]])

    return mu, theta[0]*np.eye(2)