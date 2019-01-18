#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

data = np.array([
    (1970, 165, 54.6),
    (1971, 89, 53.3),
    (1972, 55, 56.3),
    (1973, 34, 49.6),
    (1974, 9, 47.1),
    (1975, 30, 45.9),
    (1976, 59, 48.5),
    (1977, 83, 50.1),
    (1978, 109, 52.4),
    (1979, 127, 53.2),
    (1980, 153, 51.4),
    (1981, 112, 46.0),
    (1982, 80, 44.6)
])

plt.plot(data[:,0], data[:,1])
plt.plot(data[:,0], data[:,2])
plt.show()

plt.plot(data[:,1], data[:,2], '.b')
plt.show()

fit = np.polyfit(data[:,1], data[:,2], 1, full=True)
ddof = len(fit[0])
t = fit[1][0]
ts = scipy.stats.chi.ppf(0.99, len(data)-ddof)
print(t < ts)

