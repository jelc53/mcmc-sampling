import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st

from scipy.stats import multivariate_t, invgamma, beta
from common import joint_posterior_density, fetch_data
from common import generate_traceplots, generate_posterior_histograms


def sigma_sq_conditional_t12(data, mu, gam):
    """Generate sigma_sq|theta[-1] conditional distribution"""
    X, t = fetch_data(data)
    X_t12 = X[np.logical_or(t==1, t==2)]
    mean_t1 = np.repeat(mu.reshape(2,1), 4, axis=1).T
    mean_t2 = np.repeat(gam.reshape(2,1), 4, axis=1).T
    centered_data = X_t12 - np.vstack((mean_t1, mean_t2))
    scale = 0.5 * np.sum(np.power(np.linalg.norm(centered_data, ord=2, axis=1), 2))
    num_data_points = X_t12.shape[0]

    return scale, invgamma(num_data_points)


def mu_gam_marginal_t12(data):  #TODO!!
    """Generate marginal distribution for mu and gamma"""
    X, t = fetch_data(data)
    Y_bar1  = X[t==1].mean(axis=0)
    # M =     
    return multivariate_t(loc=0, shape=0, df=4)


def sample_from_proposal_dist(data):
    """Implements trial densities for each parameter"""
    X, t = fetch_data(data)

    # sample from mu, gam marginal
    mu_gam_dist = mu_gam_marginal_t12(data)
    mu, gam = mu_gam_dist.rvs()

    # sample from sigma conditional
    scale, sigma_cond_dist = sigma_sq_conditional_t12(data, mu, gam)
    sigma_sq = scale*sigma_cond_dist.rvs()

    # sample tau from beta
    tau = beta(5, 2).rvs()

    return np.array([mu, gam, tau, sigma_sq])


def compute_proposal_density(data, prop_theta, curr_theta):  #TODO!!
    """Compute probability density of our proposal distribution"""
    X, t = fetch_data(data)
    scale, sigma_cond_dist = sigma_sq_conditional_t12(data, theta)
    mu_gam_dist = mu_gam_marginal_t12(data, theta)

    return sigma_cond_dist.pdf() * (scale*mu_gam_dist.rvs())


def multistep_importance_sampling(data, n_samples, initial_position):
    """Implement importance sampling methodology: multi-step, normalized"""
    # initialization
    samples = [initial_position]
    weights = [0]

    start = time.time()
    for it in range(n_samples):

        # sample from proposal dist q
        prop_theta = sample_from_proposal_dist(data)
        samples.append(prop_theta)

        # evaluate p(theta) and q(theta) to compute weight
        p_theta = joint_posterior_density(data, prop_theta)  # trial density
        q_theta = compute_proposal_density(data, prop_theta)  # proposal
        weights.append(p_theta/q_theta)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, samples[-1]))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    return np.array(samples), np.array(weights)


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

    samples, weights = multistep_importance_sampling(data, n_samples, initial_position)
    np.save('mis_samples.npy', samples); np.save('mis_weights.npy', weights)
    # samples = np.load('../output/mis_samples.npy'); weights = np.load('../output/mis_weights.npy')
    weighted_samples = np.multiply(samples, weights) / np.sum(weights)
    generate_traceplots(weighted_samples[burn_in:], prefix='mis_')
    generate_posterior_histograms(weighted_samples[burn_in:], prefix='mis_')
