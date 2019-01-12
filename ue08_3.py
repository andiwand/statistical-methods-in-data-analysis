#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

data = np.array([(5, 9), (10, 10), (15, 14), (20, 18), (25, 22), (30, 24), (40, 29), (50, 29)])

print('(a) corr=%f' % np.corrcoef(data[:,0], data[:,1])[1,0])
plt.plot(data[:,0], data[:,1], '.')
plt.show()

fit = np.polyfit(data[:,0], data[:,1], 1, full=True)
a, b = fit[0]
print('(b) a=%f b=%f' % (a, b))
plt.plot(data[:,0], data[:,1], '.b')
plt.plot(data[:,0], a*data[:,0]+b, '-r', label='y=%fx+%f' % (a, b))
plt.show()

expect35 = a*35+b
print('(c) expectation for 35000km=%f' % expect35)

