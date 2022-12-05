import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.stats as st

from common import generate_traceplots, generate_posterior_histograms


def generate_new_sample(theta):
    """Implements trial densities for each parameter"""
    pass

def rejection_sampling(data, n_samples, initial_position, step_size):
    """Implement importance sampling methodology"""
    # initialization
    samples = [initial_position]
    weight = 

    accept_count = 0
    start = time.time()
    for it in range(n_samples):

        # ...

        # compute weight
        if np.random.uniform(0, 1) < alpha:
            samples.append(q_new)
            accept_count += 1
        else:
            samples.append(np.copy(samples[-1]))

        if it % 20 == 0:
            print('Iteration {}: {}'.format(it, samples[-1]))

    end = time.time()
    print('Time taken: {}'.format(end - start))
    print("Acceptance ratio: {}".format(accept_count/n_samples))
    return np.array(samples)


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
    path_len = 1
    m = 5
    step_size = 0.01

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)

    samples = rejection_sampling(data, n_samples, initial_position, m, step_size, path_len)
    generate_traceplots(samples[burn_in:], prefix='is_')
    generate_posterior_histograms(samples[burn_in:], prefix='is_')
