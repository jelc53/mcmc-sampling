import os
import sys
import time
import arviz

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import log_likelihood
from scipy.stats import truncnorm, norm
from common import generate_traceplots, generate_posterior_histograms


def log_transition_density(x, y, sigma):
    """Assume proposal density is a standard normal centered at curr_theta"""
    # (2*np.pi)**(-6/2) * np.linalg.det(np.eye(len(mu)))**(-0.5) * np.exp(-0.5*(x-mu).T @ np.linalg.inv(np.eye(len(mu))) @ (x-mu))
    transition = truncnorm((0-x[0])/sigma, (np.inf-x[0])/sigma,x[0], sigma).logpdf(y[0])
    transition += truncnorm((0-x[1])/sigma, (1-x[1])/sigma,x[1], sigma).logpdf(y[1])
    transition += norm(x[2], sigma).logpdf(y[2])
    transition += norm(x[3], sigma).logpdf(y[3])
    transition += norm(x[4], sigma).logpdf(y[4])
    transition += norm(x[5], sigma).logpdf(y[5])

    return transition


def proposal_sampler(curr_theta, step_size):
    """Sample from proposed distributions q centered at curr_theta"""
    # num_params = len(curr_theta)
    # while True:
    #     randomness = np.random.normal(0, 1, num_params)
    #     proposal = curr_theta + np.multiply(step_size, randomness)

    #     if proposal[0] > 0 and (proposal[1] >= 0 and proposal[1] <= 1):
    #         break
    proposed = curr_theta.copy()

    a, b = (0 - proposed[0])/step_size, (np.inf - proposed[0])/step_size
    c, d = (0 - proposed[1])/step_size, (1 - proposed[1])/step_size
    proposed[0] = truncnorm(a, b, proposed[0], step_size).rvs()
    proposed[1] = truncnorm(c, d, proposed[1], step_size).rvs()
    proposed[2] = norm(proposed[2], step_size).rvs()
    proposed[3] = norm(proposed[3], step_size).rvs()
    proposed[4] = norm(proposed[4], step_size).rvs()
    proposed[5] = norm(proposed[5], step_size).rvs()

    return proposed


def log_hastings_ratio(proposed, curr_theta, step_size, data):
    """Compute hastings ratio given curr_theta, proposed and data"""
    q_curr_prop = log_transition_density(curr_theta, proposed, sigma=step_size)
    q_prop_curr = log_transition_density(proposed, curr_theta, sigma=step_size)
    pi_curr = log_likelihood(data, curr_theta)
    pi_prop = log_likelihood(data, proposed)

    return pi_prop + q_curr_prop - pi_curr - q_prop_curr


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
        log_hratio = log_hastings_ratio(proposed, curr_theta, step_size, data)
        accept_log_prob = min(0, log_hratio)

        if np.log(np.random.uniform(0, 1)) <= accept_log_prob:
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

    step_size = 0.05
    n_samples = 700
    burn_in = 200

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    for step_size in [0.01, 0.05, 0.1, 0.5]:
        samples, accept_ratio = metropolis_hastings(data, n_samples, step_size, initial_position)
        arviz_data_format = arviz.convert_to_dataset(samples[burn_in:].reshape(1,-1,6))
        ess = arviz.ess(arviz_data_format)
        print('Acceptance ratio: {}'.format(accept_ratio))
        print('Number of effective samples: {}'.format(ess))
        print('Effective sample mean: {}'.format(ess.mean()))

    # samples, accept_ratio = metropolis_hastings(data, n_samples, step_size, initial_position)
    # np.save('mh_samples_truncnorm.npy', samples)
    samples = np.load('../output/mh_samples.npy')
    generate_traceplots(samples[burn_in:], prefix='mh_')
    generate_posterior_histograms(samples[burn_in:], prefix='mh_')
