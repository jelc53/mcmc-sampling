import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st

from common import center_data, negative_log_prob, joint_posterior_density
from common import generate_traceplots, generate_posterior_histograms


def dVdq(data, theta):
    """Compute gradient wrt theta"""
    X = np.array(data[['X1', 'X2']])
    # t = np.array(data[['group']]).flatten()
    n, k = X.shape

    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    mu = np.array([mu1, mu2])
    gam = np.array([gam1, gam2])

    centered_data = np.array(center_data(mu.reshape(1,-1), gam.reshape(1,-1), tau, data))
    sum_sq_cd = np.trace(centered_data @ centered_data.T)
    grad = np.zeros_like(theta)

    # gradient wrt sigma_sq
    grad[0] = (n + 1) / sigma_sq - (1/(2*sigma_sq**2))*sum_sq_cd  # np.sum(np.power(np.linalg.norm(centered_data, ord=2, axis=1), 2))

    # gradient wrt tau
    grad[1] = (1/sigma_sq) * np.sum(centered_data[16:] @ (gam - mu))

    # gradient wrt mu
    grad[2:4] = -(np.sum((centered_data[0:4]), axis=0) +
                  np.sum((centered_data[8:16]) * (0.5), axis=0) +
                  np.sum((centered_data[16:]) * (tau), axis=0)
                  ) / sigma_sq

    # gradient wrt gam
    grad[4:6] = -(np.sum(centered_data[4:8], axis=0) +
                  np.sum(centered_data[8:16] * (0.5), axis=0) +
                  np.sum(centered_data[16:] * (1-tau), axis=0)
                  ) / sigma_sq

    return grad


def q_update(q, p, M_mat, step_size):
    """Helper function to update theta within bounds"""
    q_prop = q + step_size * np.linalg.inv(M_mat) @ p

    # check bounds
    if q_prop[0] < 0:
        p[0] = -p[0]
        q_prop[0] = -q_prop[0]

    if q_prop[1] < 0:
        p[1] = -p[1]
        q_prop[1] = -q_prop[1]

    if q_prop[1] > 1:
        p[1] = -p[1]
        q_prop[1] = 1 - (q_prop[1] - 1)

    return q_prop, p


def leapfrog(q, p, data, M_mat, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo"""
    q, p = np.copy(q), np.copy(p)  # curr_theta = q

    p -= step_size * dVdq(data, q) / 2
    for _ in range(int(path_len / step_size)):
        q, p = q_update(q, p, M_mat, step_size)
        p -= step_size * dVdq(data, q)

    q, p = q_update(q, p, M_mat, step_size)
    p -= step_size * dVdq(data, q) / 2

    # momentum flip at end
    return q, -p


def hamiltonian_monte_carlo(data, n_samples, initial_position, m, step_size, path_len=1):
    """Implement MH sampling methodology"""
    # initialization
    samples = [initial_position]
    size = (n_samples,) + initial_position.shape[:1]
    M_mat = np.eye(len(initial_position)) * m
    momentum = st.norm(0, m)

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
            M_mat=M_mat,
            path_len=path_len,
            step_size=step_size,
        )

        # check metropolis acceptance criterion
        # curr_log_prob = negative_log_prob(data, samples[-1]) + (0.5/m)*np.linalg.norm(p0, 2)**2
        # new_log_prob = negative_log_prob(data, q_new) + (0.5/m)*np.linalg.norm(p_new, 2)**2
        # ratio = np.exp(-new_log_prob) / np.exp(-curr_log_prob)
        curr = joint_posterior_density(data, samples[-1]) * np.exp(-(0.5/m)*np.linalg.norm(p0,2)**2)
        prop = joint_posterior_density(data, q_new) * np.exp(-(0.5/m)*np.linalg.norm(p_new,2)**2)
        alpha = min(1, prop/curr)
        print(curr, prop, alpha)

        if np.random.uniform(0, 1) < alpha:
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
    path_len = 1
    m = 5
    step_size = 0.01
    initial_position = np.array([1., 0.5, -1.25, -0.5, -0.25, 0.3])

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = hamiltonian_monte_carlo(data, n_samples, initial_position, m, step_size, path_len)
    generate_traceplots(samples[burn_in:], prefix='hmc_')
    generate_posterior_histograms(samples[burn_in:], prefix='hmc_')
