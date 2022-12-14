import os
import math

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal


def log_normal(x, mu=0, sigma=1):

    mu = np.ones_like(x) * mu
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)

    return np.sum(np.log(numerator/denominator))


# plot autocorrelation
def plot_acorr(xt, nlags=2500, prefix='mh_'):
    theta_labels = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']

    plt.subplots(figsize=(12, 6))

    for i in range(xt.shape[1]):
        plt.plot(sm.tsa.acf(xt[:,i], nlags=nlags), alpha=0.75, label=theta_labels[i])

    plt.legend()
    plt.xlabel('lag')
    # plt.title('auto-correlation')
    outfile = prefix + 'autocorr_plot.png'
    plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    plt.show(); plt.close()


def prepare_histogram_sample_data(theta, w):
    """Helper function to prepare bins for histogram"""
    n = 100
    bins = np.stack([
        np.linspace(0, 0.5, n),
        np.linspace(0, 1, n),
        np.linspace(-2, 0, n),
        np.linspace(-2, 0, n),
        np.linspace(-1, 1, n),
        np.linspace(-1, 1, n)]).T
    i = 10
    j = 2

    vals = np.zeros([n, theta.shape[1]])
    for j in range(theta.shape[1]):
        for i in range(n-1):
            vals[i,j] += [np.sum(((theta[:,j] >= bins[i,j]) & (theta[:,j] < bins[i+1,j])*1) * w)]

    return vals, bins


def generate_posterior_barcharts(vals, bins, prefix=''):
    """Create and save marginal histograms for each parameter from sampled posterior"""
    param_names = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    n=100

    # for j in range(bins.shape[1]):
    #     ax[j].bar(x=bins[:,j], height=vals[:,j], width=(bins[:,j].max() - bins[:,j].min())/n)
    #     ax[j].set_xlabel(param_names[j])

    ax[0,0].bar(x=bins[:,0], height=vals[:,0], width=(bins[:,0].max() - bins[:,0].min())/n)
    ax[0,0].set_xlabel(param_names[0])

    ax[1,0].bar(x=bins[:,1], height=vals[:,1], width=(bins[:,1].max() - bins[:,1].min())/n)
    ax[1,0].set_xlabel(param_names[1])

    ax[0,1].bar(x=bins[:,2], height=vals[:,2], width=(bins[:,2].max() - bins[:,2].min())/n)
    ax[0,1].set_xlabel(param_names[2])

    ax[1,1].bar(x=bins[:,3], height=vals[:,3], width=(bins[:,3].max() - bins[:,3].min())/n)
    ax[1,1].set_xlabel(param_names[3])

    ax[0,2].bar(x=bins[:,4], height=vals[:,4], width=(bins[:,4].max() - bins[:,4].min())/n)
    ax[0,2].set_xlabel(param_names[4])

    ax[1,2].bar(x=bins[:,5], height=vals[:,5], width=(bins[:,5].max() - bins[:,5].min())/n)
    ax[1,2].set_xlabel(param_names[5])

    plt.tight_layout()
    outfile = prefix + 'sampled_histogram.png'
    plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    plt.show(); plt.close()


def generate_traceplots(samples, prefix=''):
    """Create and save traceplots for each parameter from sampling algorithm"""
    n_samples = len(samples)-1
    param_names = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(12, 6))

    # for j in range(len(samples[0])):
    #     marginal_samples = [samples[i][j] for i in range(len(samples))]
    #     ax[j].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples, s=0.75)
    #     ax[j].set_ylabel(param_names[j])

    marginal_samples = [samples[i][0] for i in range(len(samples))]
    ax[0,0].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples, s=2)
    ax[0,0].set_xlabel(param_names[0])

    marginal_samples = [samples[i][1] for i in range(len(samples))]
    ax[1,0].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples, s=2)
    ax[1,0].set_xlabel(param_names[1])

    marginal_samples = [samples[i][2] for i in range(len(samples))]
    ax[0,1].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples, s=2)
    ax[0,1].set_xlabel(param_names[2])

    marginal_samples = [samples[i][3] for i in range(len(samples))]
    ax[1,1].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples, s=2)
    ax[1,1].set_xlabel(param_names[3])

    marginal_samples = [samples[i][4] for i in range(len(samples))]
    ax[0,2].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples, s=2)
    ax[0,2].set_xlabel(param_names[4])

    marginal_samples = [samples[i][5] for i in range(len(samples))]
    ax[1,2].scatter(np.linspace(0, n_samples, num=len(samples)), marginal_samples, s=2)
    ax[1,2].set_xlabel(param_names[5])

    plt.xlabel('number of samples')
    plt.tight_layout()
    outfile = prefix + 'sampled_traceplot.png'
    plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    plt.show(); plt.close()


def generate_posterior_histograms(samples, prefix=''):
    """Create and save marginal histograms for each parameter from sampled posterior""" 
    param_names = ['sigma_sq', 'tau', 'mu1', 'mu2', 'gam1', 'gam2']
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    # for j in range(len(samples[0])):
    #     marginal_samples = [samples[i][j] for i in range(len(samples))]
    #     ax[j].hist(marginal_samples, bins=50)
    #     ax[j].set_xlabel(param_names[j])

    marginal_samples = [samples[i][0] for i in range(len(samples))]
    ax[0,0].hist(marginal_samples, bins=50)
    ax[0,0].set_xlabel(param_names[0])

    marginal_samples = [samples[i][1] for i in range(len(samples))]
    ax[1,0].hist(marginal_samples, bins=50)
    ax[1,0].set_xlabel(param_names[1])

    marginal_samples = [samples[i][2] for i in range(len(samples))]
    ax[0,1].hist(marginal_samples, bins=50)
    ax[0,1].set_xlabel(param_names[2])

    marginal_samples = [samples[i][3] for i in range(len(samples))]
    ax[1,1].hist(marginal_samples, bins=50)
    ax[1,1].set_xlabel(param_names[3])

    marginal_samples = [samples[i][4] for i in range(len(samples))]
    ax[0,2].hist(marginal_samples, bins=50)
    ax[0,2].set_xlabel(param_names[4])

    marginal_samples = [samples[i][5] for i in range(len(samples))]
    ax[1,2].hist(marginal_samples, bins=50)
    ax[1,2].set_xlabel(param_names[5])

    plt.tight_layout()
    outfile = prefix + 'sampled_histogram.png'
    plt.savefig(os.path.join(os.path.pardir, 'output', outfile))
    plt.show(); plt.close()


def log_normal(x, mu=0, sigma=1):
    """Helper function to query pdf from a log std normal"""
    mu = np.ones_like(x) * mu
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)

    return np.sum(np.log(numerator/denominator))


def negative_log_prob(data, theta):
    """Compute negative log probability evaluated at a given theta"""
    n, k = data[['X1', 'X2']].shape
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    mu = np.array([mu1, mu2]).reshape(1,-1)
    gam = np.array([gam1, gam2]).reshape(1,-1)

    # center data
    centered_data = center_data(mu, gam, tau, data)

    # compute log likelihood
    norm_term = (n*k*0.5) * np.log(2*math.pi)
    sigma_term = (n + 1) * np.log(sigma_sq)
    posterior_term = np.sum(np.power(np.linalg.norm(centered_data, ord=2, axis=1), 2))
    output = norm_term + sigma_term + (0.5/sigma_sq)*posterior_term

    return output


def log_likelihood(data, theta):
    """Compute log likelihood given data and params"""
    sigma_sq, tau, mu1, mu2, gam1, gam2 = theta
    X, t = fetch_data(data)
    mu = [mu1, mu2]
    gam = [gam1, gam2]

    means = np.zeros_like(X)
    means[t == 1] = mu
    means[t == 2] = gam
    means[t == 3] = [x + y for x, y in zip([0.5 * x for x in mu], [0.5 * x for x in gam])]
    means[t == 4] = [x + y for x, y in zip([tau * x for x in mu], [(1 - tau) * x for x in gam])]

    pdfs = [multivariate_normal(mean, sigma_sq * np.eye(2)) for mean in means]

    likelihoods = [dist.logpdf(element) for dist, element in zip(pdfs, X)]

    return np.sum(likelihoods) - np.log(sigma_sq)


def joint_posterior_density(data, theta):
    """Compute posterior density given params theta and data"""
    pi_list = []
    n_groups = len(data['group'].unique())

    for i in range(1, n_groups+1):
        mu, sigma = fetch_param_for_group(theta, group_id=i)
        x_grp = fetch_data_for_group(data, group_id=i)
        n_data_points = x_grp.shape[0]

        for j in range(n_data_points):
            x = np.array(x_grp[j:j+1])
            dist = multivariate_normal(mean=mu, cov=sigma)
            pi_list.append(dist.pdf(x))

    return np.prod(pi_list)*(1/theta[0])


def fetch_data_for_group(data, group_id):
    """Helper function to fetch gene expressions within specified group"""
    filtered_data = data[data['group'] == group_id].copy()
    return filtered_data[['X1', 'X2']]


def fetch_param_for_group(theta, group_id):
    """Helper function to fetch parameters for specified gene group"""
    if group_id == 1:
        mean = np.array([theta[2], theta[3]])

    if group_id == 2:
        mean = np.array([theta[4], theta[5]])

    if group_id == 3:
        mean = 0.5*np.array([theta[2], theta[3]]) + 0.5*np.array([theta[4], theta[5]])

    if group_id == 4:
        mean = theta[1]*np.array([theta[2], theta[3]]) + (1-theta[1])*np.array([theta[4], theta[5]])

    return mean, theta[0]*np.eye(2)


def fetch_data(data):
    """Helper function to fetch data and group vector"""
    X = np.array(data[['X1', 'X2']])
    t = np.array(data[['group']]).flatten()
    return X, t


def center_data(mu, gam, tau, data):
    """Apply hierarchical logic to center data given theta"""
    means_a = np.repeat(mu, 4, axis=0)
    means_b = np.repeat(gam, 4, axis=0)
    means_c = 0.5 * np.repeat(mu, 8, axis=0) + 0.5 * np.repeat(gam, 8, axis=0)
    means_d = tau * np.repeat(mu, 8, axis=0) + (1 - tau) * np.repeat(gam, 8, axis=0)
    means = np.vstack((means_a, means_b, means_c, means_d))
    centered_data = data[['X1', 'X2']] - means

    return centered_data


def beta_reparameterization(mu, sigma_sq):
    """Reparameterize beta in terms of mean and variance"""
    alpha = (((1-mu)/sigma_sq) - (1/mu))*(mu**2)
    beta = alpha*((1/mu) - 1)
    return alpha, beta
