import os
import sys
import time
import arviz

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import multivariate_normal
from common import joint_posterior_density
from common import generate_traceplots, generate_posterior_histograms


def transition_density(x, mu, sigma):
    """Assume proposal density is a standard normal centered at curr_theta"""
    # (2*np.pi)**(-6/2) * np.linalg.det(np.eye(len(mu)))**(-0.5) * np.exp(-0.5*(x-mu).T @ np.linalg.inv(np.eye(len(mu))) @ (x-mu))
    dist = multivariate_normal(mean=mu, cov=sigma*np.eye(len(mu)))
    return dist.pdf(x)


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
        accept_prob = min(1, h_ratio)

        if np.random.uniform(0, 1) <= accept_prob:
            curr_theta = proposed
            accept_count += 1

        samples.append(curr_theta)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, curr_theta))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    return np.array(samples), accept_count/it


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

    step_size = np.ones(6)*0.05
    n_samples = 5000
    burn_in = 200

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    # for step_size in [np.ones(6)*0.01, np.ones(6)*0.05, np.ones(6)*0.1, np.ones(6)*0.5]:
    #     samples, accept_ratio = metropolis_hastings(data, n_samples, step_size, initial_position)
    #     arviz_data_format = arviz.convert_to_dataset(samples[burn_in:].reshape(1,-1,6))
    #     ess = arviz.ess(arviz_data_format)
    #     print('Acceptance ratio: {}'.format(accept_ratio))
    #     print('Number of effective samples: {}'.format(ess))
    #     print('Effective sample mean: {}'.format(ess.mean()))
        # np.save('mh_samples.npy', samples)
    samples = np.load('../output/mh_samples.npy')
    generate_traceplots(samples[burn_in:], prefix='mh_')
    generate_posterior_histograms(samples[burn_in:], prefix='mh_')
