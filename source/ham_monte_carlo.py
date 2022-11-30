import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st

from scipy.stats import multivariate_normal
from common import center_data, joint_posterior_density, negative_log_prob
from common import fetch_data_for_group, fetch_param_for_group
from common import generate_traceplots, generate_posterior_histograms


def dVdq(data, theta):
    """xx"""
    return 0


def leapfrog(q, p, data, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo"""
    q, p = np.copy(q), np.copy(p)  # curr_theta = q

    p -= step_size * dVdq(data, q) / 2  # half step
    for _ in range(int(path_len / step_size) - 1):
        q += step_size * p  # whole step
        p -= step_size * dVdq(data, q)  # whole step
    q += step_size * p  # whole step
    p -= step_size * dVdq(data, q) / 2  # half step

    # momentum flip at end
    return q, -p


def gradient(theta, data):
    """
    Evaluates the gradient of log posterior likelihood at the given value of theta.
    Args: 
        theta: 7 dimensional vector of parameters.
        data: N*2 dimensional matrix of data.
    Returns:
        posterior: Value of posterior evaluated at the given value of theta.
    """
    # _n, _k = data.shape
    # sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    # mu = np.array([mu1, mu2])
    # gam = np.array([gam1, gam2])

    # _sigma_sq = theta[0]
    # _sigma = np.sqrt(_sigma_sq)
    # _lambda = 0.5
    # _tau = theta[1]
    # _mu = theta[2:4].reshape(1, -1)  # [1, 2]
    # _gamma = theta[4:].reshape(1, -1)

    # # Means
    # centered_data = np.array(center_data(_mu, _gamma, _tau, data))
    # sum_sq_cd = np.trace(centered_data @ centered_data.T)

    # # gradient
    # gradient = np.zeros_like(theta)

    # # The powers are 0.5 and 1.5 because our parameter is sigma_sq while in the equation it's sigma
    # gradient[0] = (_n + 1) / _sigma) + sum_sq_cd / (_sigma ** 4)
    # gradient[1] = np.sum(centered_data[16:] @ (_mu - _gamma).T)
    # gradient[2:4] = (np.sum((centered_data[0:4]), axis=0) +
    #                  np.sum((centered_data[8:16]) * (_lambda), axis=0) +
    #                  np.sum((centered_data[16:]) * (_tau), axis=0)
    #                  ) / _sigma_sq
    # gradient[4:] = (np.sum(centered_data[4:8], axis = 0) +
    #                 np.sum(centered_data[8:16] * (1 - _lambda), axis=0) +
    #                 np.sum(centered_data[16:] * (1 - _tau), axis=0)
    #                 ) / _sigma_sq

    return 0


def hamiltonian_monte_carlo(data, n_samples, initial_position, step_size, path_len=1):
    """Implement MH sampling methodology"""
    # initialization
    samples = [initial_position]
    size = (n_samples,) + initial_position.shape[:1]
    momentum = st.norm(0, 1)

    it = 0
    accept_count = 0
    start = time.time()
    for p0 in momentum.rvs(size=size):
        it += 1

        # integrate over our path to get new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            p0,
            data,
            path_len=path_len,
            step_size=step_size,
        )

        # check metropolis acceptance criterion
        curr_log_prob = negative_log_prob(data, samples[-1]) - np.sum(momentum.logpdf(p0))
        new_log_prob = negative_log_prob(data, q_new) - np.sum(momentum.logpdf(p_new))
        if np.log(np.random.rand()) < curr_log_prob - new_log_prob:
            samples.append(q_new)
            accept_count += 1
        else:
            samples.append(np.copy(samples[-1]))

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, samples[-1]))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    print("Acceptance ratio: {}".format(accept_count/it))
    return np.array(samples)


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(42)
    n_samples = 5000
    burn_in = 200
    path_len = 10
    step_size = 0.1
    initial_position = np.array([1, 0.5, 0, 0, 0, 0])

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = hamiltonian_monte_carlo(data, n_samples, initial_position, step_size, path_len)
    generate_traceplots(samples[burn_in:], prefix='gibbs_')
    generate_posterior_histograms(samples[burn_in:], prefix='gibbs_')
