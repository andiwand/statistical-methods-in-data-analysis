#!/usr/bin/env python3

""" Skeleton file, das eure Lernkurve beschleunigen soll,
    fuer Uebung 1.6.  
"""

## Importieren der wichtigsten Module fuer die Uebungsaufgabe.
import numpy, scipy.misc, math, scipy.integrate

## scipy.misc.comb ( n, k )  ## n ueber k = n! / ( k! (n-k)! )
## math.sin ( math.pi * x ) ## sin ( pi * x )

## lambda Funktionen erlauben Definitionen in einer einzigen Zeile
## prior = lambda p: p**2 * ( 1 - p )**3  ## prior(.5) = .03125
B = lambda k, p, n: scipy.misc.comb(n, k) * p**k * (1-p)**(n-k)
f1 = lambda p: p**2 * (1-p)**3
f2 = lambda p: math.sin(math.pi * p)**2

## Integral von f ueber das Intervall (a,b)
## scipy.integrate.quad ( f, a, b ) 

## Teile das Intervall (a,b) in n "bins"
## numpy.linspace ( a, b, n )

g1 = lambda p, k, n: B(k, p, n) * f1(p) / scipy.integrate.quad(lambda x: B(k, x, n) * f1(x), 0, 1)[0]
g2 = lambda p, k, n: B(k, p, n) * f2(p) / scipy.integrate.quad(lambda x: B(k, x, n) * f2(x), 0, 1)[0]

e1 = scipy.integrate.quad(lambda x: x * g1(x, 343, 1000), 0, 1)[0]
e2 = scipy.integrate.quad(lambda x: x * g2(x, 343, 1000), 0, 1)[0]

v1 = scipy.integrate.quad(lambda x: (x-e1)**2 * g1(x, 343, 1000), 0, 1)[0]
v2 = scipy.integrate.quad(lambda x: (x-e2)**2 * g2(x, 343, 1000), 0, 1)[0]

print("f1 -> %f %f" %(e1, v1))
print("f2 -> %f %f" %(e2, v2))

x = numpy.linspace(0, 1, 100)
y1 = numpy.vectorize(lambda x: g1(x, 343, 1000))(x)
y2 = numpy.vectorize(lambda x: g2(x, 343, 1000))(x)

## Plotten, siehe https://matplotlib.org/
from matplotlib import pyplot as plt

plt.plot(x, numpy.vectorize(f1)(x), label="f1")
plt.plot(x, numpy.vectorize(f2)(x), label="f2")
plt.show()

## Plotte y versus x, als Linie, mit Label "label1"
plt.plot(x, y1, label="g1")
plt.plot(x, y2, label="g2")
plt.title("UE01_1.6")

## Beschriftung der x-Achse
plt.xlabel("p")

## Legende
plt.legend()

## Speichere in Bilddatei (pdf, png, ... )
#plt.savefig ( "dummy.pdf" )
plt.show()

