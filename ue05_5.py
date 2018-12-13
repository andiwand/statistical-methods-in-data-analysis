#!/usr/bin/env python3

import numpy as np
import pandas as pd 
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

N = 500
n = 400

mu = np.array([2, 1])
cov = np.array([[0.8, 0.2], [0.2, 0.6]])

def data_to_x(mu, cov):
    n = len(mu)
    return np.concatenate((mu, cov[np.triu_indices(n)]))

def x_to_data(x):
    n = int(-3./2 + np.sqrt(9./4 + 2*len(x)))
    mu = x[:n]
    cov = np.zeros((n, n))
    cov[np.triu_indices(n)] = x[n:]
    cov += np.triu(cov, k=1).T
    return mu, cov

def ll1(samples, mu, v_inv):
    det_v_inv = np.linalg.det(v_inv)
    if det_v_inv <= 0: return -np.inf
    v = np.linalg.inv(v_inv)
    result = np.sum(np.log(np.array([multivariate_normal.pdf(x, mean=mu, cov=v) for x in samples])))
    return result

def ll2(samples, mu, v_inv):
    det_v_inv = np.linalg.det(v_inv)
    if det_v_inv <= 0: return -np.inf
    result = len(samples)*np.log(det_v_inv) - np.sum(np.array([np.linalg.multi_dot([(x-mu).T, v_inv, (x-mu)]) for x in samples]))
    return result

def ll3(samples, mu, v_inv):
    det_v_inv = np.linalg.det(v_inv)
    if det_v_inv <= 0: return -np.inf
    result = len(samples)*np.log(det_v_inv)
    y = samples - mu
    result += -np.einsum('ij,jk,ik', y, v_inv, y)
    return result

def minimize_ll_fun_gen(samples):
    ll = ll3
    def min_ll(x):
        mu, cov_inv = x_to_data(x)
        result = -ll(samples, mu, cov_inv)
        return result
    return min_ll

mu0 = np.array([0, 0])
cov0 = np.array([[1, 0], [0, 1]])
x0 = data_to_x(mu0, np.linalg.inv(cov0))
method = 'SLSQP'
options = { 'ftol': 1E-5 }

guesses = []
for i in range(N):
    print(i)
    samples = np.random.multivariate_normal(mu, cov, n)
    
    fun = minimize_ll_fun_gen(samples)
    opt = minimize(fun, x0, method=method, options=options)
    if not opt.success:
        print('optimize failed')
        continue
    if np.linalg.norm(opt.x[0:2]) > 10E3:
        print('outlier')
        continue
    guess_mu, guess_cov_inv = x_to_data(opt.x)
    guess_cov = np.linalg.inv(guess_cov_inv)
    guesses.append(data_to_x(guess_mu, guess_cov))
guesses = np.array(guesses)

df = pd.DataFrame(guesses)
df.columns = ['x', 'y', 'var_x', 'cov_xy', 'var_y']
df.to_csv('guesses.csv', index=False)

print('count', len(df.index))
print('mean')
print(df.mean())
print('std')
print(df.std())

plt.hist(df['x'])
plt.show()

plt.hist(df['var_x'])
plt.show()

