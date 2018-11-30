#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

m=5000
n=100
p=4
b=1
a=0.05

min=1E-4
max=1E1
points=1E3
dt=(max-min)/(points-1)
x = np.linspace(min, max, points)

post = lambda t, n, s, p: np.power(t, -n+p-1)*np.exp(-s/t-t)

count = 0
total = 0
for _ in range(m):
    t = np.random.beta(p, b)
    s = np.sum(np.random.exponential(t, n))
    
    tpost_ = lambda t: post(t, n, s, p)
    
    y_tpost_ = np.vectorize(tpost_)(x)
    y_tpost = y_tpost_ / (np.sum(y_tpost_) * dt)
    y_tpost_cdf = np.cumsum(y_tpost) * dt
    g1 = np.searchsorted(y_tpost_cdf, a/2, side='left')
    g2 = np.searchsorted(y_tpost_cdf, 1-a/2, side='right')
    if x[g1] <= t <= x[g2]:
        count += 1
    total += 1
    
    """
    c = scipy.integrate.quad(tpost_, 0, np.inf)[0]
    tpost = lambda t: tpost_(t) / c
    tb = scipy.integrate.quad(lambda t: t*tpost(t), 0, np.inf)[0]
    if np.isnan(tb):
        continue # quickfix
    tpost_cdf = lambda t: scipy.integrate.quad(lambda t_: tpost(t_), 0, t)[0]
    g1 = scipy.optimize.fsolve(lambda t: tpost_cdf(t)-a/2, tb)[0]
    g2 = scipy.optimize.fsolve(lambda t: tpost_cdf(t)-1+a/2, tb)[0]
    print(g1, t, g2)
    if g1 <= t <= g2:
        count +=1
    total += 1
    """

print(count/total)

