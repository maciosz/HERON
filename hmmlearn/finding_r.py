#!/usr/bin/python

import copy
import mynumpy as np

from scipy.stats import nbinom
from scipy.special import digamma

# dl / dr = sum_t digamma(x_t + r) * post - sum_t digamma(r) * post + sum_t ln(1-p) * post

def calculate_derivative(pstwa, dane, r, p):
    def _digamma(array):
        return digamma(array.astype("float64")).astype("float128")
    n_comp, n_var = r.shape
    n_obs, n_var = dane.shape
    r_conc = np.concatenate([r] * n_obs, axis=0)
    r_conc = r_conc.reshape(n_obs, n_comp, n_var)
    X_repeat = np.repeat(dane, n_comp, axis=0)
    X_repeat = X_repeat.reshape(n_obs, n_comp, n_var)
    suma = X_repeat + r_conc
    suma = suma.reshape(n_obs, n_comp, n_var)
    pstwa_repeat = np.repeat(pstwa, n_var)
    pstwa_repeat = pstwa_repeat.reshape(n_obs, n_comp, n_var)
    a = np.sum(pstwa_repeat * _digamma(suma), axis=0)
    a = a.reshape(n_comp, n_var)
    b = np.sum(pstwa_repeat * _digamma(r_conc), axis=0)
    b = b.reshape(n_comp, n_var)
    p_conc = np.concatenate([p] * n_obs, axis=0)
    p_conc = p_conc.reshape(n_obs, n_comp, n_var)
    c = np.sum(pstwa_repeat * np.log(1 - p_conc), axis=0)
    c = c.reshape(n_comp, n_var)
    derivative = a - b + c
    return derivative

def update_r(r, derivative, delta):
    n_comp, n_var = r.shape
    for i in xrange(n_comp):
        for j in xrange(n_var):
            if abs(derivative[i, j]) <= 1e-10:
                continue
            if delta[i, j] == 0:
                if derivative[i, j] < 0:
                    delta[i, j] = r[i, j] * -0.5
                elif derivative[i, j] > 0:
                    delta[i, j] = r[i, j] * 10 + 100
                else:
                    print "cos nie tak, pewnie nan"
            elif delta[i, j] < 0:
                if derivative[i, j] < 0:
                    pass
                elif derivative[i, j] > 0:
                    delta[i, j] *= -0.5
                else:
                    print "cos nie tak, pewnie nan"
            elif delta[i, j] > 0:
                if derivative[i, j] < 0:
                    delta[i, j] *= -0.5
                elif derivative[i, j] > 0:
                    pass
                else:
                    print "cos nie tak, pewnie nan"
            r[i, j] += delta[i, j]
            if r[i, j] <= 0:
                if derivative[i, j] < 0:
                    print "r bliskie zero, ale pochodna ujemna..."
                r[i, j] = 0.0001
    return r, delta

def find_r(r_initial, dane, pstwa, p, threshold = 5e-2):
    r = r_initial.copy()
    r_not_found = True
    #p = 1.0 - p
    #counter = 0
    delta = np.zeros(shape=r.shape)
    while r_not_found:
        derivative = calculate_derivative(pstwa, dane, r, p)
        #r_test = np.repeat(r[0], r.shape[0])[:, np.newaxis]
        #derivative_test = calculate_derivative(pstwa, dane, r_test, p)
        #if derivative[0] != derivative_test[0]:
        #    print "sa rozne..."
        if np.all(abs(derivative) < threshold):
            r_not_found = False
            break
        r, delta = update_r(copy.deepcopy(r), derivative, delta)
        #counter += 1
        #if counter % 10 == 0:
        #    print "%i iterations" % counter
        #    #print derivative[0]
    return r

