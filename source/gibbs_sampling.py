import os
import sys
import time

import numpy as np
import pandas as pd

from scipy.stats import truncnorm, invgamma
from common import fetch_data, fetch_data_for_group, fetch_param_for_group
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
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    mu = np.array([mu1, mu2])
    gam = np.array([gam1, gam2])

    # check bounds
    if (gam - mu).any() == 0:
        return tau

    # compute mean, std
    yi_4 = fetch_data_for_group(data, group_id=4)
    n_4 = yi_4.shape[0]

    denom = n_4*(np.linalg.norm(mu-gam, 2)**2)
    mean_vec = yi_4 @ mu - gam @ gam - yi_4 @ gam - mu @ gam
    mean = np.sum(mean_vec) / denom
    std = sigma_sq / denom

    # sample truncated normal
    bounds = (0, 1)
    a, b = (bounds[0] - mean) / std, (bounds[1] - mean) / std
    dist = truncnorm(a, b, loc=mean, scale=std)
    # print(mean, std, dist.rvs())

    return dist.rvs()


def sample_mu_conditional_posterior(data, theta):
    """Sample from mu|theta[-1] conditional posterior"""
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    # mu = np.array([mu1, mu2])
    gam = np.array([gam1, gam2])

    # fetch data
    X, t = fetch_data(data)
    y_1, y_3, y_4 = X[t==1], X[t==3], X[t==4]
    n_1, n_3, n_4 = len(y_1), len(y_3), len(y_4)

    # group 1
    mean_1 = y_1
    alpha_1 = 1

    # group 3
    # mean_3 = 0.5*y_3 - 0.25*gam
    mean_3 = 2*y_3 - gam
    alpha_3 = 0.5**2

    # group 4
    # mean_4 = 0.25*tau*y_4 - 0.5*tau*(1-tau)*gam
    mean_4 = (y_4 - (1-tau)*gam) / tau
    alpha_4 = tau**2

    # compute mean, std
    w_134 = alpha_1*n_1 + alpha_3*n_3 + alpha_4*n_4
    mean = np.sum([alpha_1*np.sum(mean_1, axis=0), alpha_3*np.sum(mean_3, axis=0), alpha_4*np.sum(mean_4, axis=0)], axis=0) / w_134
    cov = sigma_sq / w_134 * np.eye(2)

    return np.random.multivariate_normal(mean, cov)


def sample_gam_conditional_posterior(data, theta):
    """Sample from gam|theta[-1] conditional posterior"""
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    mu = np.array([mu1, mu2])
    # gam = np.array([gam1, gam2])

    # fetch data
    X, t = fetch_data(data)
    y_2, y_3, y_4 = X[t==2], X[t==3], X[t==4]
    n_2, n_3, n_4 = len(y_2), len(y_3), len(y_4)

    # group 2
    mean_2 = y_2
    alpha_2 = 1

    # group 3
    mean_3 = 0.5*y_3 - 0.25*mu
    alpha_3 = 0.5**2

    # group 4
    mean_4 = 0.25*tau*y_4 - 0.5*tau*(1-tau)*mu
    alpha_4 = tau**2

    # compute mean, std
    w_234 = alpha_2*n_2 + alpha_3*n_3 + alpha_4*n_4
    mean = np.sum([alpha_2*np.sum(mean_2, axis=0), alpha_3*np.sum(mean_3, axis=0), alpha_4*np.sum(mean_4, axis=0)], axis=0) / w_234
    cov = sigma_sq / w_234 * np.eye(2)

    return np.random.multivariate_normal(mean, cov)


def gibbs_sampling(data, n_samples, initial_position):
    """Implement MH sampling methodology"""
    samples = [initial_position]

    it = 0
    start = time.time()
    while it < n_samples:  # num_samples * num_params
        it += 1
        # print(curr_theta)
        curr_theta = samples[-1].copy()
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

        elif idx == 3:
            # sample gamma from its normal distribution
            gam_proposal = sample_gam_conditional_posterior(data, curr_theta)
            curr_theta[4:6] = gam_proposal

        else:
            print("Error: trying to update out-of-bounds parameter!")

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
    initial_position = np.array([
        1.,  # sigma
        0.5,  # tau
        -1.25,  # mu1
        -0.5,  # mu2
        -0.25,  # gam1
        0.3  # gam2
    ])

    n_samples = 5000
    burn_in = 200

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = gibbs_sampling(data, n_samples, initial_position)
    generate_traceplots(samples[burn_in:], prefix='gibbs_')
    generate_posterior_histograms(samples[burn_in:], prefix='gibbs_')
