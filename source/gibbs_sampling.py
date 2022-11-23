import os
import sys

import numpy as np
import pandas as pd

from scipy.stats import invgamma
# from common import fetch_data_for_group, fetch_param_for_group
from common import fetch_data_for_group, generate_traceplots, generate_posterior_histograms


def sample_tau_conditional_posterior(data, theta):
    """Sample from tau|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    # compute mean, std
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean = np.sum(np.divide(yi_4 - gam, (mu - gam)))
    std = sigma_sq / np.sum(gam - mu)

    # check bounds
    if std <= 0:
        return tau

    return np.random.normal(mean, std)


def sample_mu_conditional_posterior(data, theta):
    """Sample from mu|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    # group 1
    yi_1 = fetch_data_for_group(data, group_id=1)
    mu_1 = yi_1

    # group 3
    yi_3 = fetch_data_for_group(data, group_id=3)
    mu_3 = (yi_3 - 0.5*gam) / 0.5

    # group 4
    yi_4 = fetch_data_for_group(data, group_id=4)
    mu_4 = (yi_4 - 1 + tau*gam) / tau

    # compute mean, std
    denom = 1*4 + 0.5**2*8 + tau**2*8  # TODO: not sure about this
    mean = (np.sum(mu_1) + np.sum(mu_3) + np.sum(mu_4)) / denom
    std = sigma_sq / denom

    # check bounds
    if std <= 0:
        return mu

    return np.random.normal(mean, std)


def sample_gam_conditional_posterior(data, theta):
    """Sample from gam|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    # group 2
    yi_2 = fetch_data_for_group(data, group_id=2)
    gam_2 = yi_2

    # group 3
    yi_3 = fetch_data_for_group(data, group_id=3)
    gam_3 = (yi_3 - 0.5*mu) / 0.5

    # group 4
    yi_4 = fetch_data_for_group(data, group_id=4)
    gam_4 = (yi_4 - tau*mu) / (1 - tau)

    # compute mean, std
    denom = 1*4 + (0.5-1)**2*8 + (tau-1)**2*8  # TODO: not sure about this
    mean = (np.sum(gam_2) + np.sum(gam_3) + np.sum(gam_4)) / denom
    std = sigma_sq / denom

    # check bounds
    if std <= 0:
        return gam

    return np.random.normal(mean, std)


def gibbs_sampling(data, n_samples):
    """Implement MH sampling methodology"""
    num_data_points = data.shape[0]
    current = [1, 0.5, 0, 0, 0, 0]
    samples = [current]

    it = 0
    while it < n_samples:  # num_samples * num_params
        it += 1
        idx = it % 4  # systematic

        if idx == 0:
            # sample sigma_sq from invgamma(n+2)
            sigma_sq_proposal = invgamma.rvs(num_data_points + 2)
            current[0] = sigma_sq_proposal

        elif idx == 1:
            # sample tau from [insert conjugate] distribution
            tau_proposal = sample_tau_conditional_posterior(data, current)
            current[1] = tau_proposal

        elif idx == 2:
            # sample mu from its normal distribution
            mu_proposal = sample_mu_conditional_posterior(data, current)
            current[2:4] = mu_proposal

        else:
            # sample gamma from its normal distribution
            gam_proposal = sample_gam_conditional_posterior(data, current)
            current[4:6] = gam_proposal

        samples.append(current)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, current))

    print("Gibbs sampling accpetance ratio: {}".format(it/it))
    return samples


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(42)
    # step_size = np.ones(6)*0.05
    n_samples = 1000
    burn_in = 200

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = gibbs_sampling(data, n_samples)
    generate_traceplots(samples[burn_in:], prefix='gibbs_')
    generate_posterior_histograms(samples[burn_in:], prefix='gibbs_')
