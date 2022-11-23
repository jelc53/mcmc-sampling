import os
import sys

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from common import fetch_data_for_group, fetch_param_for_group
from common import generate_traceplots, generate_posterior_histograms


def transition_density(x, mu, sigma):
    """Assume proposal density is a standard normal centered at current"""
    # (2*np.pi)**(-6/2) * np.linalg.det(np.eye(len(mu)))**(-0.5) * np.exp(-0.5*(x-mu).T @ np.linalg.inv(np.eye(len(mu))) @ (x-mu))
    dist = multivariate_normal(mean=mu, cov=sigma*np.eye(len(mu)))
    return dist.pdf(x)


def posterior_density(data, theta):
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


def hastings_ratio(proposed, current, data):
    """Compute hastings ratio given current, proposed and data"""
    q_curr_prop = transition_density(current, proposed, sigma=1)
    q_prop_curr = transition_density(proposed, current, sigma=1)
    pi_curr = posterior_density(data, current)
    pi_prop = posterior_density(data, proposed)

    return (pi_prop/pi_curr) * (q_curr_prop/q_prop_curr)


def beta_reparameterization(mu, sigma_sq):
    """Reparameterize beta in terms of mean and variance"""
    alpha = (((1-mu)/sigma_sq) - (1/mu))*(mu**2)
    beta = alpha*((1/mu) - 1)
    return alpha, beta


def proposal_sampler(current):
    """Sample from proposed distributions q centered at current"""
    # proposed = [np.random.normal(current[i]) for i in range(n_params)]
    sigma_sq = np.random.normal(np.sqrt(current[0]))**2
    alpha, beta = beta_reparameterization(current[1], 0.005)
    tau = np.random.beta(alpha, beta)
    mu1 = np.random.normal(current[2])
    mu2 = np.random.normal(current[3])
    gam1 = np.random.normal(current[4])
    gam2 = np.random.normal(current[5])

    return [sigma_sq, tau, mu1, mu2, gam1, gam2]


def metropolis_hastings(data, n_samples):
    """Implement MH sampling methodology"""
    current = [1, 0.5, 0, 0, 0, 0]
    samples = [current]

    it = 0; accept_count = 0
    while it < n_samples:
        it += 1

        proposed = proposal_sampler(current)
        h_ratio = hastings_ratio(proposed, current, data)
        # print(h_ratio)
        accept_prob = min(1, h_ratio)

        if np.random.uniform(0, 1) <= accept_prob:
            current = proposed
            accept_count += 1

        samples.append(current)
        # print(current)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, current))

    print("Metropolis-Hastings accpetance ratio: {}".format(accept_count/it))
    return samples


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(0)
    n_samples = 100

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)
    samples = metropolis_hastings(data, n_samples)

    generate_traceplots(samples, prefix='mh_')
    generate_posterior_histograms(samples, prefix='mh_')
