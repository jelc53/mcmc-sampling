import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st

from autograd import grad
from scipy.stats import multivariate_normal
from common import center_data, log_normal, negative_log_likelihood
from common import fetch_data_for_group, fetch_param_for_group
from common import generate_traceplots, generate_posterior_histograms


def update_theta(eps, M, phi, curr_theta):
    proposal = curr_theta.reshape(-1,1) + eps * np.linalg.inv(M) @ phi
    # Boundary checks
    if proposal[0] >= 0 and 0 <= proposal[1] <= 1 and 0 <= proposal[2] <= 1:
        theta = proposal
    else:
        theta = curr_theta
    return theta


def update_phi(phi, eps, theta, data, step_ratio=1):
    return phi + step_ratio * eps * get_gradient_log_posterior(theta, data).reshape(-1,1)


def get_gradient_log_posterior(theta, data):
    """
    Evaluates the gradient of log posterior likelihood at the given value of theta.
    Args: 
        theta: 7 dimensional vector of parameters.
        data: N*2 dimensional matrix of data.
    Returns:
        posterior: Value of posterior evaluated at the given value of theta.
    """
    _n, _k = data.shape

    _sigma_sq = theta[0]
    _sigma = np.sqrt(_sigma_sq)
    _lambda = 0.5
    _tau = theta[1]
    _mu = theta[2:4].reshape(1, -1)  # [1, 2]
    _gamma = theta[4:].reshape(1, -1)

    # Means
    centered_data = np.array(center_data(_mu, _gamma, _tau, data))
    sum_sq_cd = np.trace(centered_data @ centered_data.T)

    # gradient
    gradient = np.zeros_like(theta)

    # The powers are 0.5 and 1.5 because our parameter is sigma_sq while in the equation it's sigma
    gradient[0] = (-2 * (_n + 1) / _sigma) + sum_sq_cd / (_sigma ** 3)
    gradient[1] = np.sum(centered_data[16:] @ (_mu - _gamma).T)
    gradient[2:4] = (np.sum((centered_data[0:4]), axis=0) +
                     np.sum((centered_data[8:16]) * (_lambda), axis=0) +
                     np.sum((centered_data[16:]) * (_tau), axis=0)
                     ) / _sigma_sq
    gradient[4:] = (np.sum(centered_data[4:8], axis = 0) +
                    np.sum(centered_data[8:16] * (1 - _lambda), axis=0) +
                    np.sum(centered_data[16:] * (1 - _tau), axis=0)
                    ) / _sigma_sq

    return gradient


def hamiltonian_monte_carlo(data, n_samples, step_size, initial_position):
    """Implement MH sampling methodology"""
    num_params = np.shape(initial_position)[0]
    curr_theta = initial_position.copy()
    samples = [initial_position]
    num_leapfrogs = 10
    M = np.eye(num_params)

    it = 0
    accepted = 0
    start = time.time()
    while it < n_samples:  # num_samples * num_params
        it += 1

        phi = st.multivariate_normal.rvs(mean=0, cov=1, size=[num_params,1]).reshape(-1,1)
        curr_phi = phi  # phi_{t-1}
        theta = curr_theta  # theta_{t-1}

        for step in range(num_leapfrogs):
            if step == 0 or step == num_leapfrogs - 1:
                # half update phi
                phi = update_phi(phi, step_size, theta, data, step_ratio=0.5)

                # full update theta
                theta = update_theta(step_size, M, phi, theta)

                # half update phi
                phi = update_phi(phi, step_size, theta, data, step_ratio=0.5)
            else:
                # full update theta, phi
                theta = update_theta(step_size, M, phi, theta)
                phi = update_phi(phi, step_size, theta, data, step_ratio=1)

        # At the end of leap frog steps, phi* = phi and theta* = theta
        # Compute the acceptance probability
        curr_posterior = get_joint_log_posterior(curr_theta, data) + log_normal(curr_phi)
        prop_posterior = get_joint_log_posterior(theta, data)  + log_normal(phi)

        # - because operating in log scale
        ratio_posterior = prop_posterior - curr_posterior
        alpha = min(0, ratio_posterior)  # Since np.log(1) = 0
        alpha = np.exp(alpha)  # To convert from log-scale to normal-scale

        # Whether to accept proposal with probability alpha
        if np.random.uniform(0, 1) < alpha:
            curr_theta = theta
            accepted += 1
        else:
            curr_theta = curr_theta

        print(curr_theta)
        samples.append(curr_theta)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, curr_theta))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    print("Acceptance ratio: {}".format(accepted/it))
    return samples


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(42)
    n_samples = 5000
    burn_in = 200
    leapfrogs = 10
    step_size = 1.0/leapfrogs
    initial_position = np.array([1, 0.5, 0, 0, 0, 0])

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = hamiltonian_monte_carlo(data, n_samples, step_size, initial_position)
    generate_traceplots(samples[burn_in:], prefix='gibbs_')
    generate_posterior_histograms(samples[burn_in:], prefix='gibbs_')
