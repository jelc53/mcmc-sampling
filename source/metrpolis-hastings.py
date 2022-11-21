import os
import sys
import scipy

import pandas as pd
import numpy as np


def multivariate_normal_pdf(x, mu, sigma):
    """Scipy stats package"""
    return scipy.stats.multivariate_normal(x, mu, sigma)

def metropolis_hastings(data):
    n = data.shape[0]
    for i in range(n):
        x = 
        tau = np.random.uniform(0,1)
    pass


if __name__ == '__main__':

    infile = sys.argv[1]

    file_path = os.path.join(os.path.pardir, 'data', infile)
    data = pd.read_csv(file_path)
    metropolis_hastings(data)
