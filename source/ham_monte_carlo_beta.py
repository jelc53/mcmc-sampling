import os
import sys
import time
import arviz

import numpy as np
import pandas as pd
import scipy.stats as st
# from autograd import grad

from common import center_data, fetch_data, log_likelihood, negative_log_prob
from common import generate_traceplots, generate_posterior_histograms


def dVdq(data, theta):
    """Compute gradient wrt theta"""
    X, t = fetch_data(data)
    n, k = X.shape

    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    mu = np.array([mu1, mu2])
    gam = np.array([gam1, gam2])

    centered_data = np.array(center_data(mu.reshape(1,-1), gam.reshape(1,-1), tau, data))
    sum_sq_cd = np.trace(centered_data @ centered_data.T)
    grad = np.zeros_like(theta)

    # gradient wrt sigma_sq
    grad[0] = (n + 1) / sigma_sq - (1/(2*sigma_sq**2))*sum_sq_cd

    # gradient wrt tau
    beta_term = (1-20)/tau + 2/(1-tau)
    grad[1] = beta_term + (1/sigma_sq) * np.sum(centered_data[16:], axis=0) @ (gam - mu)

    # gradient wrt mu
    grad[2:4] = -(np.sum(centered_data[0:4], axis=0) +
                  0.5*np.sum(centered_data[8:16], axis=0) +
                  tau*np.sum(centered_data[16:], axis=0)
                  ) / sigma_sq

    # gradient wrt gam
    grad[4:6] = -(np.sum(centered_data[4:8], axis=0) +
                  0.5*np.sum(centered_data[8:16], axis=0) +
                  (1-tau)*np.sum(centered_data[16:], axis=0)
                  ) / sigma_sq

    return grad


def q_update(q, p, M_mat, step_size):
    """Helper function to update theta within bounds"""
    q_prop = q + step_size * np.linalg.inv(M_mat) @ p

    # check bounds
    # if q_prop[0] < 0:
    #     p[0] = -p[0]
    #     q_prop[0] = -q_prop[0]

    # if q_prop[1] > 1:
    #     p[1] = -p[1]
    #     q_prop[1] = -q_prop[1]

    # if q_prop[1] < 0:
    #     p[1] = -p[1]
    #     q_prop[1] = 1 - (q_prop[1]-1)

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
        curr_u = - log_likelihood(data, samples[-1]) - st.beta(a=20, b=3).logpdf(samples[-1][1])
        prop_u = - log_likelihood(data, q_new) - st.beta(a=20, b=3).logpdf(q_new[1])
        curr_k = (0.5/m)*np.linalg.norm(p0,2)**2
        prop_k = (0.5/m)*np.linalg.norm(p_new,2)**2
        log_hratio = curr_u - prop_u + curr_k - prop_k
        accept_log_prob = min(0, log_hratio)

        if np.log(np.random.uniform(0, 1)) <= accept_log_prob:
            samples.append(q_new)
            accept_count += 1
        else:
            samples.append(np.copy(samples[-1]))

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, samples[-1]))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    return np.array(samples), accept_count/it


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(42)
    initial_position = np.array([
        1.,  # sigma
        0.8,  # tau
        -1.25,  # mu1
        -0.5,  # mu2
        -0.25,  # gam1
        0.3  # gam2
    ])

    n_samples = 5000
    burn_in = 200
    path_len = 1
    m = 1
    step_size = 0.01

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    # for step_size in [0.005, 0.01, 0.05]:
    #     for m in [1, 5, 10]:
    #         samples, accept_ratio = hamiltonian_monte_carlo(data, n_samples, initial_position, m, step_size, path_len)
    #         arviz_data_format = arviz.convert_to_dataset(samples[burn_in:].reshape(1,-1,6))
    #         ess = arviz.ess(arviz_data_format)
    #         print('Acceptance ratio: {}'.format(accept_ratio))
    #         print('Number of effective samples: {}'.format(ess))
    #         print('Effective sample mean: {}'.format(ess.mean()))
            # np.save('hmc_samples.npy', samples)
    samples, accept_ratio = hamiltonian_monte_carlo(data, n_samples, initial_position, m, step_size, path_len)
    np.save('hmc_samples.npy', samples)
    # samples = np.load('../output/hmc_samples.npy')
    generate_traceplots(samples[burn_in:], prefix='hmc_')
    generate_posterior_histograms(samples[burn_in:], prefix='hmc_')
