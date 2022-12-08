import os
import sys
import time
import arviz
import math

import autograd.numpy as np
import pandas as pd
import scipy.stats as st
from autograd import grad

from common import joint_posterior_density
from common import generate_traceplots, generate_posterior_histograms


def center_data_autograd(mu, gam, tau, data):
    """Apply hierarchical logic to center data given theta"""

    means_a = np.repeat(mu.reshape(1,-1), 4, axis=0)
    means_b = np.repeat(gam.reshape(1,-1), 4, axis=0)
    means_c = 0.5 * np.repeat(mu.reshape(1,-1), 8, axis=0) + 0.5 * np.repeat(gam.reshape(1,-1), 8, axis=0)
    means_d = tau * np.repeat(mu.reshape(1,-1), 8, axis=0) + (1 - tau) * np.repeat(gam.reshape(1,-1), 8, axis=0)
    means = np.vstack((means_a, means_b, means_c, means_d))
    centered_data = data[['X1', 'X2']] - means

    return centered_data


def negative_log_prob_autograd(theta):
    """Compute negative log probability evaluated at a given theta"""
    file_path = os.path.join(os.path.pardir, 'data', 'data.csv')
    data = pd.read_csv(file_path)

    n, k = data[['X1', 'X2']].shape
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta._value
    mu = np.array([mu1, mu2])
    gam = np.array([gam1, gam2])

    # center data
    centered_data = center_data_autograd(mu, gam, tau, data)

    # compute log likelihood
    norm_term = (n*k*0.5) * np.log(2*math.pi)
    sigma_term = (n + 1) * np.log(sigma_sq)
    posterior_term = np.sum(np.power(np.linalg.norm(centered_data, ord=2, axis=1), 2))
    output = norm_term + sigma_term + (0.5/sigma_sq)*posterior_term

    return output


# def dVdq(data, theta):
#     """Compute gradient wrt theta"""
#     X = np.array(data[['X1', 'X2']])
#     # t = np.array(data[['group']]).flatten()
#     n, k = X.shape

#     sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
#     mu = np.array([mu1, mu2])
#     gam = np.array([gam1, gam2])

#     centered_data = np.array(center_data(mu.reshape(1,-1), gam.reshape(1,-1), tau, data))
#     sum_sq_cd = np.trace(centered_data @ centered_data.T)
#     grad = np.zeros_like(theta)

#     # means = np.zeros_like(X)
#     # means[t == 1] = mu
#     # means[t == 2] = gam
#     # means[t == 3] = [x + y for x, y in zip([0.5 * x for x in mu], [0.5 * x for x in gam])]
#     # means[t == 4] = [x + y for x, y in zip([tau * x for x in mu], [(1 - tau) * x for x in gam])]

#     # gradient wrt sigma_sq
#     grad[0] = (n + 1) / sigma_sq - (1/(2*sigma_sq**2))*sum_sq_cd  # np.sum(np.power(np.linalg.norm(centered_data, ord=2, axis=1), 2))

#     # gradient wrt tau
#     grad[1] = (1/sigma_sq) * np.sum(centered_data[16:], axis=0) @ (gam - mu)

#     # gradient wrt mu
#     grad[2:4] = -(np.sum((centered_data[0:4]), axis=0) +
#                   np.sum((centered_data[8:16]) * (0.5), axis=0) +
#                   np.sum((centered_data[16:]) * (tau), axis=0)
#                   ) / sigma_sq

#     # gradient wrt gam
#     grad[4:6] = -(np.sum(centered_data[4:8], axis=0) +
#                   np.sum(centered_data[8:16] * (0.5), axis=0) +
#                   np.sum(centered_data[16:] * (1-tau), axis=0)
#                   ) / sigma_sq

#     return grad


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
        q_prop[1] = 1 - (q_prop[1]-1)

    return q_prop, p


def leapfrog(q, p, dVdq, M_mat, path_len, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo"""
    q, p = np.copy(q), np.copy(p)  # curr_theta = q

    p -= step_size * dVdq(q) / 2
    for _ in range(int(path_len / step_size)):
        q, p = q_update(q, p, M_mat, step_size)
        p -= step_size * dVdq(q)

    q, p = q_update(q, p, M_mat, step_size)
    p -= step_size * dVdq(q) / 2

    # momentum flip at end
    return q, -p


def hamiltonian_monte_carlo(data, n_samples, initial_position, m, step_size, path_len=1):
    """Implement MH sampling methodology"""
    # initialization
    samples = [initial_position]
    size = (n_samples,) + initial_position.shape[:1]
    M_mat = np.eye(len(initial_position)) * m
    momentum = st.norm(0, m)

    # autograd magic
    dVdq = grad(negative_log_prob_autograd)

    it = 0
    accept_count = 0
    start = time.time()
    for p0 in momentum.rvs(size=size):
        it += 1

        # integrate over our path to get new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            p0,
            dVdq,
            M_mat=M_mat,
            path_len=path_len,
            step_size=step_size,
        )

        # check metropolis acceptance criterion
        curr = joint_posterior_density(data, samples[-1]) * np.exp(-(0.5/m)*np.linalg.norm(p0,2)**2)
        prop = joint_posterior_density(data, q_new) * np.exp(-(0.5/m)*np.linalg.norm(p_new,2)**2)
        alpha = min(1, prop/curr)
        # print(curr, prop, alpha)

        if np.random.uniform(0, 1) < alpha:
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
        0.5,  # tau
        -1.25,  # mu1
        -0.5,  # mu2
        -0.25,  # gam1
        0.3  # gam2
    ])

    n_samples = 500
    burn_in = 200
    path_len = 1
    m = 5
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
    np.save('hmc_autograd_samples.npy', samples)
    samples = np.load('../output/hmc_autograd_samples.npy')
    generate_traceplots(samples[burn_in:], prefix='hmc_')
    generate_posterior_histograms(samples[burn_in:], prefix='hmc_')
