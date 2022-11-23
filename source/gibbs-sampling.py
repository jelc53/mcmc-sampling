import os
import sys

import numpy as np
import pandas as pd

# from scipy.stats import multivariate_normal
# from common import fetch_data_for_group, fetch_param_for_group
from common import generate_traceplots, generate_posterior_histograms


def beta_reparameterization(mu, sigma_sq):
    """Reparameterize beta in terms of mean and variance"""
    alpha = (((1-mu)/sigma_sq) - (1/mu))*(mu**2)
    beta = alpha*((1/mu) - 1)
    return alpha, beta


def proposal_sampler(current, index):
    """Sample from proposed distributions q centered at current"""
    # proposed = [np.random.normal(current[i]) for i in range(n_params)]
    proposed = current.copy()
    if index == 0:
        proposed[index] = np.random.normal(np.sqrt(current[index]))**2

    elif index == 1:
        alpha, beta = beta_reparameterization(current[index], 0.001)
        proposed[index] = np.random.beta(alpha, beta)

    else:
        proposed[index] = np.random.normal(current[index])

    return proposed


def gibbs_sampling(data, n_samples):
    """Implement MH sampling methodology"""
    current = [1, 0.5, 0, 0, 0, 0]
    samples = [current]

    it = 0; accept_count = 0
    while it < n_samples:
        it += 1
        i = it % 6

        proposed = proposal_sampler(current, index=i)

        if True:  # will always meet acceptance criteria!
            current = proposed
            accept_count += 1
        samples.append(current)

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, current))

    print("Gibbs sampling accpetance ratio: {}".format(accept_count/it))
    return samples


if __name__ == '__main__':

    infile = sys.argv[1]

    np.random.seed(0)
    n_samples = 1000

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)
    samples = gibbs_sampling(data, n_samples)

    generate_traceplots(samples, prefix='gibbs_')
    generate_posterior_histograms(samples, prefix='gibbs_')
