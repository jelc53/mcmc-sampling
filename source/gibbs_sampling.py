import os
import sys
import time

import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy.stats import invgamma
# from common import fetch_data_for_group, fetch_param_for_group
from common import fetch_data_for_group, fetch_param_for_group, generate_traceplots, generate_posterior_histograms


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
    if (gam - mu).all() == 0:
        return tau

    # compute mean, std
    n_4 = data[['X1', 'X2']]
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean = np.array(np.divide((yi_4 - gam), (mu - gam)))
    alpha = np.sum([np.sum(mean[i]) for i in range(len(mean))])
    std = (sigma_sq / np.linalg.norm(gam-mu, 2)**2)

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
    std_3 = (sigma_sq / 0.5**2) * np.eye(2)
    multi_dim_samples_3 = np.array([np.random.multivariate_normal(mean_3[i], std_3) for i in range(yi_3.shape[0])])
    multi_dim_samples.append(multi_dim_samples_3)

    # group 4
    yi_4 = fetch_data_for_group(data, group_id=4)
    mean_4 = np.array((yi_4 - (1-tau)*gam) / tau)
    std_4 = (sigma_sq / tau**2) * np.eye(2)
    multi_dim_samples_4 = np.array([np.random.multivariate_normal(mean_4[i], std_4) for i in range(yi_4.shape[0])])
    multi_dim_samples.append(multi_dim_samples_4)

    # compute mean, std
    # denom = 1*4 + 0.5**2*8 + tau**2*8  # TODO: not sure about this!
    # mean = (np.sum(mean_1, axis=0) + np.sum(mean_3, axis=0) + np.sum(mean_4, axis=0)) / denom
    # std = (sigma_sq / denom) * np.eye(2)

    return np.prod(np.vstack(multi_dim_samples), axis=0)
    # return np.random.multivariate_normal(mean, std)


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
    # mean = (np.sum(mean_2, axis=0) + np.sum(mean_3 * (0.5-1)**2, axis=0) + np.sum(mean_4 * (tau-1)**2, axis=0)) / denom
    # std = (sigma_sq / denom) * np.eye(2)

    return np.prod(np.vstack(multi_dim_samples), axis=0)
    # return np.random.multivariate_normal(mean, std)


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
    # step_size = np.ones(6)*0.05
    n_samples = 5000
    burn_in = 200

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = gibbs_sampling(data, n_samples, initial_position)
    generate_traceplots(samples[burn_in:], prefix='gibbs_')
    generate_posterior_histograms(samples[burn_in:], prefix='gibbs_')
