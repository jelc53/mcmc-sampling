import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st

from scipy.stats import multivariate_t, invgamma, beta
from common import fetch_data, prepare_histogram_sample_data, generate_posterior_barcharts


def fetch_invgam_params(data, mu, gam):
    """Generate sigma_sq|theta[-1] conditional distribution"""
    X, t = fetch_data(data)
    X_t12 = X[np.logical_or(t==1, t==2)]

    sum_sq = (np.sum(np.square(X[t==1] - mu.reshape(1,-1))) +
              np.sum(np.square(X[t==2] - gam.reshape(1,-1))))

    return X_t12.shape[0], sum_sq/2


def fetch_tdist_params(data):
    """Generate marginal distribution for mu and gamma"""
    X, t = fetch_data(data)
    n_1, n_2 = np.sum(t==1), np.sum(t==2)

    Y_bar1 = X[t==1].mean(axis=0)
    Y_bar2 = X[t==2].mean(axis=0)

    M_1 = np.sum(np.square(X[t==1]-Y_bar1))
    M_2 = np.sum(np.square(X[t==2]-Y_bar2))
    M = M_1 + M_2

    nu = 2*(n_1 + n_2) - 4
    mean = np.hstack((Y_bar1, Y_bar2))
    cov = (M / nu) * np.diag(np.array([1/n_1, 1/n_1, 1/n_2, 1/n_2]))

    return mean, cov, nu


def sample_from_proposal_dist(data):
    """Implements trial densities for each parameter"""
    # sample from mu, gam marginal
    mean, cov, nu = fetch_tdist_params(data)
    mu1, mu2, gam1, gam2 = multivariate_t.rvs(loc=mean, shape=cov, df=nu)
    mu = np.array([mu1, mu2]).reshape(-1,1)
    gam = np.array([gam1, gam2]).reshape(-1,1)

    # sample from sigma conditional
    n_12, scale = fetch_invgam_params(data, mu, gam)
    sigma_sq = invgamma.rvs(a=n_12, scale=scale)

    # sample tau from beta
    tau = beta(20, 3).rvs()

    return np.array([sigma_sq, tau, mu1, mu2, gam1, gam2])


def log_proposal_density(data, prop_theta):
    """Compute probability density of our proposal distribution"""
    _, _, mu1, mu2, gam1, gam2 = prop_theta
    mu = np.array([mu1, mu2]).reshape(-1,1)
    gam = np.array([gam1, gam2]).reshape(-1,1)

    # density evaluation @ prop_theta
    mean, cov, nu = fetch_tdist_params(data)
    n_12, scale = fetch_invgam_params(data, mu, gam)

    x_tdist = np.vstack((mu, gam)).reshape(1,-1)
    tdist_logpdf = multivariate_t.logpdf(x=x_tdist, loc=mean, shape=cov, df=nu)
    invgam_logpdf = invgamma.logpdf(x=prop_theta[0], a=n_12, scale=scale)

    return tdist_logpdf + invgam_logpdf


def compute_weight_for_sample(theta, data):
    """Compute weights for q samples as p(Y_{3,4}|theta)"""
    X, t = fetch_data(data)
    n_3, n_4 = np.sum(t==1), np.sum(t==2)
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    mu, gam = np.array([mu1, mu2]), np.array([gam1, gam2])

    # sum_sq = (np.sum(np.power(np.linalg.norm(X[t==3] - (mu + gam)/2, ord=2, axis=1),2)) +
    #           np.sum(np.power(np.linalg.norm(X[t==4] - (tau*mu + (1-tau)*gam), ord=2, axis=1),2)))
    sum_sq = (np.sum(np.square(X[t==3] - (mu + gam)/2)) +
              np.sum(np.square(X[t==4] - (tau*mu + (1-tau)*gam))))

    return (1/sigma_sq)**(n_3+n_4) * np.exp(-1/(2*sigma_sq) * sum_sq)


def multistep_importance_sampling(data, num_samples, initial_position):
    """Implement importance sampling methodology: multi-step, normalized"""
    # initialization
    samples = [initial_position]
    weights = [0]

    start = time.time()
    for it in range(num_samples):

        # sample from proposal dist q
        prop_theta = sample_from_proposal_dist(data)
        samples.append(prop_theta)

        # evaluate p(theta) and q(theta) to compute weight
        weights.append(compute_weight_for_sample(prop_theta, data))
        # p_theta = log_likelihood(data, prop_theta)  # trial density
        # q_theta = log_proposal_density(data, prop_theta)  # proposal
        # weights.append(np.exp(p_theta - q_theta))

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, samples[-1]))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    return np.array(samples[1:]), np.array(weights[1:])


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

    num_raw_samples = 1000000

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    # samples, weights_u = multistep_importance_sampling(data, num_raw_samples, initial_position)
    # np.save('mis_samples.npy', samples); np.save('mis_weights_u.npy', weights_u)
    samples = np.load('../output/mis_samples.npy'); weights_u = np.load('../output/mis_weights_u.npy')
    w = weights_u / np.sum(np.sum(weights_u))
    theta_hat = samples * np.reshape(w, (w.shape[0],1))
    variance = np.sum(np.square(samples - np.mean(samples, axis=0)) * np.reshape(w, (w.shape[0],1)), axis=0)
    print('Mean of weighted samples: {}'.format(np.sum(theta_hat, axis=0)))
    print('Variance of weighted samples: {}'.format(variance))
    # np.save('mis_weighted_samples.npy', theta_hat)

    vals, bins = prepare_histogram_sample_data(samples, w)
    generate_posterior_barcharts(vals, bins, prefix='mis_')
    # weighted_idx = np.random.choice(samples.shape[0], size=num_weighted_samples, p=weights/np.sum(weights))
    # weighted_samples = samples[weighted_idx]
