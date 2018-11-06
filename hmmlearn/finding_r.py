#!/usr/bin/python

# TO DZIALA!
# tylko nie dla wielowymiarowych.
# ani dla tych dziwnych nizej gdzie probowalam udawac ze mam jeden stan.
# cos sie z wymiarami pochrzanilo, bo wczesniej dzialalo.
# dla wielowymiarowych nie wiem nawet jak powinno byc.


import copy
import numpy as np
from scipy.stats import nbinom
from scipy.special import digamma



# dl / dr = sum_t digamma(x_t + r) * post - sum_t digamma(r) * post + sum_t ln(1-p) * post

def calculate_derivative(pstwa, dane, r, p):
    #print pstwa.T
    #print "** pstwa shape:"
    #print pstwa.shape
    #print

    #print dane
    #print "** dane shape:"
    #print dane.shape

    #print r.T
    #print "** r shape:"
    #print r.shape

    #print "** p shape:"
    #print p.shape

    n_comp, n_var = r.shape
    n_obs, n_var = dane.shape

    #suma = np.array([dane] * n_comp) + np.repeat(r[:, np.newaxis], n_obs, axis=1)
    #suma = np.squeeze(suma).T
    suma = (dane + r.T)

    #print "suma:"
    #print suma
    #print
    #print "pstwa * suma:"
    #print np.dot(pstwa.T, digamma(suma))
    a = np.diag(np.dot(pstwa.T, digamma(suma)))
    b = np.sum(pstwa * digamma(r).T, axis=0)
    c = np.sum(pstwa * np.log(1 - p).T, axis=0)
    #print a
    #print b
    #print c
    derivative = a - b + c
    if np.all(r[0] < 1.0) & np.all(derivative[0] < -80.0):
        print "cos dziwnego:"
        print r, derivative
    return derivative

def update_r(r, r_prev, derivative, r_counter):
    #print "r_prev:", r_prev
    #print "r old:", r
    for i in xrange(len(r)):
        delta = abs(r[i] - r_prev[i])
        if np.all(delta <= 1e-7):
            r_counter[i] += 1
            if r_counter[i] > 30:
                delta = 20
                r_counter[i] = 0
        if i == 0:
            print "delta:", delta
            print "derivative:", derivative[0]
            print "r:", r[0]
        if abs(derivative[i]) <= 1e-10:
            pass
        elif derivative[i] > 0:
            if delta == 0:
                r[i] = r[i] * 10
            else:
                r[i] = r[i] + delta * 0.9 #/2.0
        elif derivative[i] < 0:
            if delta == 0:
                r[i] = r[i] / 10.0
            else:
                r[i] = r[i] - delta * 0.9 #/2.0
        if r[i] < 0:
            r[i] = 1e-3
    #print "r new:", r
    if r.shape == (1,3):
        r = r.T
        print "transponuje r, wtf"
    return r, r_counter

def find_r(r_initial, dane, pstwa, p, threshold = 1e-4):
    print "pstwa shape:", pstwa.shape
    print "r shape:", r_initial.shape
    print "dane shape:", dane.shape
    print "p shape:", p.shape
    derivatives = []
    for i in np.linspace(0, 1, 100):
        i = np.array([[i], [i], [i]])
        derivatives.append(calculate_derivative(pstwa, dane, i, p)[0])
    print derivatives
    r = copy.deepcopy(r_initial)
    r_prev = copy.deepcopy(r_initial)
    r_not_found = True
    counter = 0
    r_counter = [0] * r.shape[0]
    while r_not_found:
        derivative = calculate_derivative(pstwa, dane, r, p)
        if np.all(r[0] < 1.0) & np.all(derivative[0] < -80.0):
            print "i tu tez dziwnie, zobaczmy"        
            #r_tmp = 0.54912251
            print "derivative %f:" % r[0]
            r_tmp = np.array([r[0], r[0], r[0]])
            print calculate_derivative(pstwa, dane, r_tmp, p)[0]
            print "r, derivative:"
            print r, derivative
            print "***"
        if np.all(abs(derivative) < threshold):
            r_not_found = False
        r_new, r_counter = update_r(copy.deepcopy(r), copy.deepcopy(r_prev), derivative, r_counter)
        r, r_prev = copy.deepcopy(r_new), copy.deepcopy(r)
        counter += 1
        if counter % 10 == 0:
            print "%i iterations" % counter
            #print derivative[0]
    return r


def main():
    dane = np.array([10, 15, 20, 9, 30])[:, np.newaxis]
    dane_2d = np.array([[10, 22],
                        [15, 35],
                        [20, 32],
                        [9, 15],
                        [30, 39]])
    pstwa = np.array([[0.1, 0.8, 0.9, 0.2, 0.9],
                      [0.9, 0.2, 0.1, 0.8, 0.1]]).T

    p = np.array([[0.4], [0.5]])
    r = np.array([[10.0], [20.0]])

    p_2d = np.array([[0.4, 0.5],
                     [0.7, 0.3]])
    r_2d = np.array([[10, 20],
                     [40, 30]])


    r_not_found = True
    r_prev = r

    licznik = 0

    while r_not_found:
        derivative = calculate_derivative(pstwa, dane, r, p)
        print "DERIVATIVE:", derivative
        print
        """
        derivative = calculate_derivative(pstwa.T[0].T, dane,
                                          np.array([r[0]]), np.array([p[0]]))
        print "DERIVATIVE:"
        print derivative
        print
        derivative = calculate_derivative(pstwa.T[1].T, dane,
                                          np.array( [r[1]]), np.array([p[1]]))
        print "DERIVATIVE:"
        print derivative
        """
        """
        print "2d:"
        derivative = calculate_derivative(pstwa, dane_2d, r_2d, p_2d)
        print "DERIVATIVE:"
        print derivative
        print
        derivative = calculate_derivative(pstwa.T[0].T, dane_2d, r_2d[0], p_2d[0])
        print "DERIVATIVE:"
        print derivative
        print
        derivative = calculate_derivative(pstwa.T[1].T, dane_2d, r_2d[1], p_2d[1])
        print "DERIVATIVE:"
        print derivative
        """
        if np.all(abs(derivative) < 1e-3):
            r_not_found = False
        r_new = update_r(copy.deepcopy(r), copy.deepcopy(r_prev), derivative)
        r, r_prev = copy.deepcopy(r_new), copy.deepcopy(r)
        # nie wiem czy to przypisanie zadziala, czy cos sie nie nadpisze
        licznik += 1
        #print "jestem na %i. iteracji" % licznik
        if licznik > 10000:
            break

if __name__ == '__main__':
    main()

