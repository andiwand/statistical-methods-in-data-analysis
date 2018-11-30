#!/usr/bin/env python3

import numpy as np
import scipy.optimize
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
randnorm = np.random.normal
uniform = np.random.uniform
minimize = scipy.optimize.minimize
normpdf = scipy.stats.norm.pdf

def plot_like(rand, f):
    x_mu, y_p = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(0, 1, 100))
    z_ll = np.array([f(rand, x, y) for x, y in zip(np.ravel(x_mu), np.ravel(y_p))]).reshape(x_mu.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mu, y_p, z_ll)
    ax.set_xlabel('mu')
    ax.set_ylabel('p')
    ax.set_zlabel('ll')
    plt.show()

def plot_contour(xy):
    # calculate the 2D density of the data given
    counts, xbins, ybins = np.histogram2d(xy[:,0], xy[:,1], bins=20, normed=LogNorm())
    # make the contour plot
    plt.contour(counts.transpose(), extent=(xbins.min(), xbins.max(), ybins.min(), ybins.max()), linewidths=3, colors='black', linestyles='solid')
    plt.show()

N = 5000
n = 250

mu = 0
p = 0.7
sigma1 = 1
sigma2 = 10

def loglike(x, mu, p):
    a = normpdf(x, loc=mu, scale=sigma1)
    b = normpdf(x, loc=mu, scale=sigma2)
    return np.sum(np.log(p*a+(1-p)*b))

rand = None
estimates = []
for i in range(N):
    rand_p = uniform(low=0, high=1, size=n)
    rand = randnorm(size=n) * (sigma1 + (sigma2 - sigma1) * (rand_p > p))
    x0 = np.array([0, 0.5])
    x_max = minimize(lambda x: -loglike(rand, x[0], x[1]), x0, bounds=((-5, 5), (0, 0.99)))
    if not x_max.success or np.array_equal(x_max.x, x0):
        print('error: could not find minimum')
        continue
    x_max = x_max.x
    estimates.append(x_max)
estimates = np.array(estimates)

plot_like(rand, loglike)
plot_contour(estimates)

