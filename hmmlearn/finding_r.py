#!/usr/bin/python

# nie dziala dla wielowymiarowych.
# nie wiem nawet jak powinno byc.


import copy
import numpy as np

from scipy.stats import nbinom
from scipy.special import digamma


# dl / dr = sum_t digamma(x_t + r) * post - sum_t digamma(r) * post + sum_t ln(1-p) * post

def calculate_derivative(pstwa, dane, r, p):
    n_comp, n_var = r.shape
    n_obs, n_var = dane.shape
    suma = (dane + r.T)
    a = np.diag(np.dot(pstwa.T, digamma(suma)))
    b = np.sum(pstwa * digamma(r).T, axis=0)
    c = np.sum(pstwa * np.log(1 - p).T, axis=0)
    derivative = a - b + c
    return derivative

def update_r(r, derivative, delta):
    for i in xrange(len(r)):
        if abs(derivative[i]) <= 1e-10:
            continue
        if delta[i] == 0:
            if derivative[i] < 0:
                delta[i] = r[i] * -0.5
            elif derivative[i] > 0:
                delta[i] = r[i] * 10 + 100
            else:
                print "cos nie tak, pewnie nan"
        elif delta[i] < 0:
            if derivative[i] < 0:
                pass
            elif derivative[i] > 0:
                delta[i] *= -0.5
            else:
                print "cos nie tak, pewnie nan"
        elif delta[i] > 0:
            if derivative[i] < 0:
                delta[i] *= -0.5
            elif derivative[i] > 0:
                pass
            else:
                print "cos nie tak, pewnie nan"
        r[i] += delta[i]
        if r[i] <= 0:
            if derivative[i] < 0:
                print "r bliskie zero, ale pochodna ujemna..."
            r[i] = 0.0001
    return r, delta


def find_r(r_initial, dane, pstwa, p, threshold = 5e-2):
    r = r_initial.copy()
    r_not_found = True
    p = 1.0 - p
    counter = 0
    delta = [0] * r.shape[0]
    while r_not_found:
        derivative = calculate_derivative(pstwa, dane, r, p)
        r_test = np.repeat(r[0], r.shape[0])[:, np.newaxis]
        derivative_test = calculate_derivative(pstwa, dane, r_test, p)
        if derivative[0] != derivative_test[0]:
            print "sa rozne..."
        if np.all(abs(derivative) < threshold):
            r_not_found = False
            break
        r, delta = update_r(copy.deepcopy(r), derivative, delta)
        counter += 1
        #if counter % 10 == 0:
        #    print "%i iterations" % counter
        #    #print derivative[0]
    return r
