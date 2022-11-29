import os
import sys
import time

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
from common import fetch_data_for_group, fetch_param_for_group
from common import generate_traceplots, generate_posterior_histograms


def transition_density(x, mu, sigma):
    """Assume proposal density is a standard normal centered at curr_theta"""
    # (2*np.pi)**(-6/2) * np.linalg.det(np.eye(len(mu)))**(-0.5) * np.exp(-0.5*(x-mu).T @ np.linalg.inv(np.eye(len(mu))) @ (x-mu))
    dist = multivariate_normal(mean=mu, cov=sigma*np.eye(len(mu)))
    return dist.pdf(x)


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


def hastings_ratio(proposed, curr_theta, data):
    """Compute hastings ratio given curr_theta, proposed and data"""
    q_curr_prop = transition_density(curr_theta, proposed, sigma=1)
    q_prop_curr = transition_density(proposed, curr_theta, sigma=1)
    pi_curr = joint_posterior_density(data, curr_theta)
    pi_prop = joint_posterior_density(data, proposed)

    return (pi_prop/pi_curr) * (q_curr_prop/q_prop_curr)


def proposal_sampler(curr_theta, step_size):
    """Sample from proposed distributions q centered at curr_theta"""
    # proposed = [np.random.normal(curr_theta[i]) for i in range(n_params)]
    num_params = len(curr_theta)
    while True:
        randomness = np.random.normal(0, 1, num_params)
        proposal = curr_theta + np.multiply(step_size, randomness)

        if proposal[0] > 0 and (proposal[1] >= 0 and proposal[1] <= 1):
            break
    # print(proposal)
    return proposal


def metropolis_hastings(data, n_samples, step_size, inital_position):
    """Implement MH sampling methodology"""
    curr_theta = inital_position.copy()
    samples = [inital_position]

    it = 0
    accept_count = 0
    start = time.time()
    while it < n_samples:
        it += 1

        proposed = proposal_sampler(curr_theta, step_size)
        h_ratio = hastings_ratio(proposed, curr_theta, data)
        # print(h_ratio)
        accept_prob = min(1, h_ratio)

        if np.random.uniform(0, 1) <= accept_prob:
            curr_theta = proposed
            accept_count += 1

        samples.append(curr_theta)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, curr_theta))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    print("Acceptance ratio: {}".format(accept_count/it))
    # print('Autocorrelation: {}'.format(sm.tsa.acf([samples[i][1] for i in range(len(samples))])))
    return samples


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(42)
    inital_position = [1, 0.5, 0, 0, 0, 0]
    step_size = np.ones(6)*0.05
    n_samples = 500
    burn_in = 200

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = metropolis_hastings(data, n_samples, step_size, inital_position)
    generate_traceplots(samples[burn_in:], prefix='mh_')
    generate_posterior_histograms(samples[burn_in:], prefix='mh_')
