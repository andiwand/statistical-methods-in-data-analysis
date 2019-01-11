#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

group_count = 5
tau = 3
alpha = 0.05

data = np.loadtxt('block7.txt')
print(len(data), data.min(), data.max())

data.sort()
groups = np.array_split(data, group_count)
group_counts = np.array([len(g) for g in groups])
group_edges = np.array([0] + [(g1.max()+g2.min())*0.5 for g1, g2 in zip(groups[:-1], groups[1:])] + [13])
print(group_edges)

def exp_cdf(t, b, a=0):
    return np.exp(-a/t) - np.exp(-b/t)

expected_counts = np.array([exp_cdf(tau, b, a) for a, b in zip(group_edges[:-1], group_edges[1:])]) * len(data)

chi2 = np.sum((group_counts - expected_counts)**2 / expected_counts)
print('(a) accept h0?', chi2 < scipy.stats.chi2.ppf(1-alpha, group_count - 1))

tau_guess = np.mean(data)
print('(b) tau=%f' % tau_guess)
expected_counts = np.array([exp_cdf(tau_guess, b, a) for a, b in zip(group_edges[:-1], group_edges[1:])]) * len(data)
chi2 = np.sum((group_counts - expected_counts)**2 / expected_counts)
print('(b) accept h0?', chi2 < scipy.stats.chi2.ppf(1-alpha, group_count - 2))

