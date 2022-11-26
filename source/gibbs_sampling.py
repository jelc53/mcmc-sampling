import os
import sys

import numpy as np
import pandas as pd

from scipy.stats import invgamma
# from common import fetch_data_for_group, fetch_param_for_group
from common import fetch_data_for_group, generate_traceplots, generate_posterior_histograms


def sample_sigma_sq_conditional_posterior(data, theta):
    """Sample from sigma_sq|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    num_data_points = data.shape[0]
    num_groups = len(data['group'].unique())

    mean = []
    for j in range(1,num_groups+1):
        yi_j = fetch_data_for_group(data, group_id=j)
        mean.append(np.repeat(mu.reshape(2,1), yi_j.shape[0], axis=1).T)
    scale = 0.5 * np.sum(np.power(np.linalg.norm(data[['X1', 'X2']]-np.vstack(mean), ord=2, axis=1), 2))

    return scale * invgamma.rvs(num_data_points)  # n + 2?


def sample_tau_conditional_posterior(data, theta):
    """Sample from tau|theta[-1] conditional posterior"""
    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    # check bounds
    if (gam - mu).all() == 0:
        return tau

    # compute mean, std
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean = np.array(np.divide((gam - yi_4), (mu - gam)))
    std = (sigma_sq / np.linalg.norm(gam-mu, 2)**2) * np.eye(2)

    # sampling
    multi_dim_samples = np.array([np.random.multivariate_normal(mean[i], std) for i in range(yi_4.shape[0])])
    multi_dim_joint_sample = np.prod(np.vstack(multi_dim_samples), axis=0)

    return multi_dim_joint_sample.mean()  # is this the right way to combine?


def sample_mu_conditional_posterior(data, theta):
    """Sample from mu|theta[-1] conditional posterior"""
    multi_dim_samples = []

    tau = theta[1]
    sigma_sq = theta[0]
    # mu = np.array([theta[2], theta[3]])
    gam = np.array([theta[4], theta[5]])

    # group 1
    yi_1 = fetch_data_for_group(data, group_id=1)
    mean_1 = np.array(yi_1)
    std_1 = sigma_sq * np.eye(2)
    multi_dim_samples_1 = np.array([np.random.multivariate_normal(mean_1[i], std_1) for i in range(yi_1.shape[0])])
    multi_dim_samples.append(multi_dim_samples_1)

    # group 3
    yi_3 = fetch_data_for_group(data, group_id=3)
    mean_3 = np.array(2*yi_3 - gam)
    std_3 = 4*sigma_sq* np.eye(2)
    multi_dim_samples_3 = np.array([np.random.multivariate_normal(mean_3[i], std_3) for i in range(yi_3.shape[0])])
    multi_dim_samples.append(multi_dim_samples_3)

    # group 4
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean_4 = np.array((yi_4 - (1-tau)*gam) / tau)
    std_4 = (sigma_sq / tau**2) * np.eye(2)
    multi_dim_samples_4 = np.array([np.random.multivariate_normal(mean_4[i], std_4) for i in range(yi_4.shape[0])])
    multi_dim_samples.append(multi_dim_samples_4)

    # # compute mean, std
    # denom = 1*4 + 0.5**2*8 + tau**2*8  # TODO: not sure about this
    # mean = (np.sum(mu_1) + np.sum(mu_3) + np.sum(mu_4)) / denom
    # std = sigma_sq / denom

    return np.prod(np.vstack(multi_dim_samples), axis=0)


def sample_gam_conditional_posterior(data, theta):
    """Sample from gam|theta[-1] conditional posterior"""
    multi_dim_samples = []

    tau = theta[1]
    sigma_sq = theta[0]
    mu = np.array([theta[2], theta[3]])
    # gam = np.array([theta[4], theta[5]])

    # group 2
    yi_2 = fetch_data_for_group(data, group_id=2)
    mean_2 = np.array(yi_2)
    std_2 = sigma_sq * np.eye(2)
    multi_dim_samples_2 = np.array([np.random.multivariate_normal(mean_2[i], std_2) for i in range(yi_2.shape[0])])
    multi_dim_samples.append(multi_dim_samples_2)

    # group 3
    yi_3 = fetch_data_for_group(data, group_id=3)
    mean_3 = np.array(2*yi_3 - mu)
    std_3 = 4*sigma_sq * np.eye(2)
    multi_dim_samples_3 = np.array([np.random.multivariate_normal(mean_3[i], std_3) for i in range(yi_3.shape[0])])
    multi_dim_samples.append(multi_dim_samples_3)

    # group 4
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean_4 = np.array((yi_4 - tau*mu) / (1 - tau))
    std_4 = (sigma_sq / (1-tau)**2) * np.eye(2)
    multi_dim_samples_4 = np.array([np.random.multivariate_normal(mean_4[i], std_4) for i in range(yi_4.shape[0])])
    multi_dim_samples.append(multi_dim_samples_4)

    # # compute mean, std
    # denom = 1*4 + (0.5-1)**2*8 + (tau-1)**2*8  # TODO: not sure about this
    # mean = (np.sum(gam_2) + np.sum(gam_3) + np.sum(gam_4)) / denom
    # std = sigma_sq / denom

    return np.prod(np.vstack(multi_dim_samples), axis=0)


def gibbs_sampling(data, n_samples):
    """Implement MH sampling methodology"""
    current = [1, 0.5, 0, 0, 0, 0]
    samples = [current]

    it = -1
    while it < n_samples:  # num_samples * num_params
        it += 1
        idx = it % 4  # systematic

        if idx == 0:
            # sample sigma_sq from invgamma(n+2)
            sigma_sq_proposal = sample_sigma_sq_conditional_posterior(data, current)
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

        print(current)
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
