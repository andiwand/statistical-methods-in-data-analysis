#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

l_from=0
l_to=4
l_steps=1000

n=60
S=124

poisson=lambda k,l: np.prod([np.exp(-l/max(1,k))*l/i for i in range(1, max(2,k))])
priori_=lambda l: 1/(1+0.25*l)
posteriori_=lambda l,k: poisson(k,l)*priori_(l)

x=np.linspace(l_from, l_to, l_steps)
y_priori=np.vectorize(lambda l: priori_(n*l))(x)
y_priori/=(x[1]-x[0])*np.sum(y_priori)
y_posteriori=np.vectorize(lambda l: posteriori_(n*l, S))(x)
y_posteriori/=(x[1]-x[0])*np.sum(y_posteriori)

plt.plot(x,y_priori,'b',x,y_posteriori,'r')

def hdp(x,y,gamma):
    lo,hi=0,np.max(y)
    dx=x[1]-x[0]
    last_cover=0
    while True:
        mid=0.5*(hi+lo)
        cover=dx*np.sum(y[y>=mid])
        if cover==last_cover and cover>gamma: break
        if cover>gamma: lo=mid
        else: hi=mid
        last_cover=cover
    return np.min(x[y>=mid]), np.max(x[y>=mid]), cover

interval=hdp(x,y_posteriori,0.95)
print("hdp from %f to %f covers %f" % interval)
mask=(x>=interval[0]) & (x<=interval[1])
plt.plot(x,y_posteriori,'r')
plt.fill_between(x, y_posteriori, where=mask)

