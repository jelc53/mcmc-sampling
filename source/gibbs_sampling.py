import os
import sys
import time

import numpy as np
import pandas as pd

from scipy.stats import truncnorm, invgamma
from common import fetch_data_for_group, fetch_param_for_group
from common import generate_traceplots, generate_posterior_histograms


def sample_sigma_sq_conditional_posterior(data, theta):
    """Sample from sigma_sq|theta[-1] conditional posterior"""
    num_data_points = data.shape[0]
    num_groups = len(data['group'].unique())

    mean = []
    for j in range(1, num_groups+1):
        yi_j = fetch_data_for_group(data, group_id=j)
        mean_j, cov_j = fetch_param_for_group(theta, group_id=j)
        mean.append(np.repeat(mean_j.reshape(2,1), yi_j.shape[0], axis=1).T)
    scale = 0.5 * np.sum(np.power(np.linalg.norm(data[['X1', 'X2']]-np.vstack(mean), ord=2, axis=1), 2))

    return scale * invgamma.rvs(num_data_points)  # n + 2?


def sample_tau_conditional_posterior(data, theta):
    """Sample from tau|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    # check bounds
    if (gam - mu).any() == 0:  # all()?
        return tau

    # compute mean, std
    n_4 = data[data['group'] == 4].shape[0]
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean_vec = np.array(np.divide((yi_4 - gam), (mu - gam)))
    mean = np.sum([np.sum(mean_vec[i]) for i in range(len(mean_vec))]) / n_4
    std = (sigma_sq / (n_4*np.linalg.norm(gam-mu, 2)**2))

    # sample truncated normal
    a = (0-mean) / std
    b = (1-mean) / std
    proposal_tau = truncnorm.rvs(a, b, loc=mean, scale=std)

    return proposal_tau


def sample_mu_conditional_posterior(data, theta):
    """Sample from mu|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    # mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    # group 1
    yi_1 = fetch_data_for_group(data, group_id=1)
    mean_1 = np.array(yi_1)
    n_1 = len(yi_1)
    alpha_1 = 1

    # group 3
    yi_3 = fetch_data_for_group(data, group_id=3)
    mean_3 = np.array(2*yi_3 - gam)
    n_3 = len(yi_3)
    alpha_3 = 0.5**2

    # group 4
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean_4 = np.array((yi_4 - (1-tau)*gam) / tau)
    n_4 = len(yi_4)
    alpha_4 = tau**2

    # compute mean, std
    w_134 = alpha_1*n_1 + alpha_3*n_3 + alpha_4*n_4
    mean = np.sum([alpha_1*np.sum(mean_1, axis=0), alpha_3*np.sum(mean_3, axis=0), alpha_4*np.sum(mean_4, axis=0)], axis=0) / w_134
    std = sigma_sq / w_134 * np.eye(2)

    return np.random.multivariate_normal(mean, std)


def sample_gam_conditional_posterior(data, theta):
    """Sample from gam|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    # gam = np.array([theta[4], theta[5]])

    # group 2
    yi_2 = fetch_data_for_group(data, group_id=1)
    mean_2 = np.array(yi_2)
    n_2 = len(yi_2)
    alpha_2 = 1

    # group 3
    yi_3 = fetch_data_for_group(data, group_id=3)
    mean_3 = np.array((yi_3 - 0.5*mu)/0.5)
    n_3 = len(yi_3)
    alpha_3 = 0.5**2

    # group 4
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean_4 = np.array((yi_4 - tau*mu) / (1-tau))
    n_4 = len(yi_4)
    alpha_4 = (1-tau)**2

    # compute mean, std
    w_234 = alpha_2*n_2 + alpha_3*n_3 + alpha_4*n_4
    mean = np.sum([alpha_2*np.sum(mean_2, axis=0), alpha_3*np.sum(mean_3, axis=0), alpha_4*np.sum(mean_4, axis=0)], axis=0) / w_234
    std = sigma_sq / w_234 * np.eye(2)

    return np.random.multivariate_normal(mean, std)


def gibbs_sampling(data, n_samples, initial_position):
    """Implement MH sampling methodology"""
    curr_theta = initial_position.copy()
    samples = [initial_position]

    it = 0
    while it < n_samples:  # num_samples * num_params
        it += 1
        start = time.time()
        idx = (it-1) % 4  # systematic

        if idx == 0:
            # sample sigma_sq from invgamma(n+2)
            sigma_sq_proposal = sample_sigma_sq_conditional_posterior(data, curr_theta)
            curr_theta[0] = sigma_sq_proposal

        elif idx == 1:
            # sample tau from [insert conjugate] distribution
            tau_proposal = sample_tau_conditional_posterior(data, curr_theta)
            curr_theta[1] = tau_proposal

        elif idx == 2:
            # sample mu from its normal distribution
            mu_proposal = sample_mu_conditional_posterior(data, curr_theta)
            curr_theta[2:4] = mu_proposal

        else:
            # sample gamma from its normal distribution
            gam_proposal = sample_gam_conditional_posterior(data, curr_theta)
            curr_theta[4:6] = gam_proposal

        print(curr_theta)
        samples.append(curr_theta)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, curr_theta))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    print("Acceptance ratio: {}".format(it/it))
    # print('Autocorrelation: {}'.format(sm.tsa.acf([samples[i][1] for i in range(len(samples))])))
    return samples


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(42)
    initial_position = [1, 0.5, 0, 0, 0, 0]
    n_samples = 5000
    burn_in = 200

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = gibbs_sampling(data, n_samples, initial_position)
    generate_traceplots(samples[burn_in:], prefix='gibbs_')
    generate_posterior_histograms(samples[burn_in:], prefix='gibbs_')
