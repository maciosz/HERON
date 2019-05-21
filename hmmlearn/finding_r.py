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
    #c = np.sum(pstwa_repeat * np.log(p_conc), axis=0)
    c = c.reshape(n_comp, n_var)
    derivative = a - b + c
    """
    if np.any(r <= 0.1):
        print "r mniejsze rowne 0.1 przy liczeniu pochodnej"
        print "r:", r
        #print "digamma(r_conc):", _digamma(r_conc)
        #print "pstwa:", pstwa
        print "p:", p
        print "b:", b
        print "derivative:", derivative 
    else:
        print "allright"
    """
    return derivative

def update_r(r, derivative, delta, stop):
    # mozna by jakos sprytniej skakac
    # np teraz mam ciag ktory skacze od kilkudziesieciu milionow
    #  na plusie i minusie (dwa razy minus, raz plus, i tak w kolko),
    #  zakres sie zaciesnia, ale powoli
    #  moze mozna by jakos uzaleznic od tego na ile duza co do modulu
    #  byla poprzednia i jeszcze poprzednia pochodna
    # albo jakos wazyc rejony czy co
    # najgorzej jak np jest caly czas ta sama dodatnia
    #  przeplatana dwoma ujemnymi, coraz blizszymi zera, ale wciaz odleglymi
    #  czlowiek by zobaczyl ze mozna tu skakac szybciej,
    #  zamiast ladowac w tym samym miejscu
    #  wiec powinno sie to tez dac zaimplementowac...
    # inna sprawa ze dla poczatkowych iteracji,
    #  kiedy estymacja p jest tez bardzo zgrubna,
    #  nie potrzebuje chyba bardzo precyzyjnej estymacji r;
    #  wystarczy mi nieduza pochodna, niekoniecznie bardzo bliska zero.
    #  Ale to juz bardziej skomplikowane do zaimplementowania.
    n_comp, n_var = r.shape
    for i in xrange(n_comp):
        for j in xrange(n_var):
            if stop[i, j] == True:
                continue
            if abs(derivative[i, j]) <= 1e-10:
                continue
            if delta[i, j] == 0:
                if derivative[i, j] < 0:
                    delta[i, j] = r[i, j] * -0.5
                elif derivative[i, j] > 0:
                    delta[i, j] = r[i, j] * 10 + 5
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
                    #print "r bliskie zero, ale pochodna ujemna..."
                    #print "obczaj: %f %f %f" % (r[i, j], derivative[i, j], delta[i, j])
                    stop[i, j] = True
                r[i, j] = 0.1
                delta[i, j] = 0
    return r, delta, stop

def find_r(r_initial, dane, pstwa, p, threshold=5e-2):
    r = r_initial.copy()
    r_not_found = True
    p = 1.0 - p
    counter = 0
    delta = np.zeros(shape=r.shape)
    stop = np.zeros(r.shape, dtype=bool)
    while r_not_found:
        derivative = calculate_derivative(pstwa, dane, copy.deepcopy(r), p)
        #print derivative
        #r_test = np.repeat(r[0], r.shape[0])[:, np.newaxis]
        #derivative_test = calculate_derivative(pstwa, dane, r_test, p)
        #if derivative[0] != derivative_test[0]:
        #    print "sa rozne..."
        if np.all(abs(derivative) < threshold + stop):
            r_not_found = False
            break
        r, delta, stop = update_r(copy.deepcopy(r), derivative, delta, stop)
        counter += 1
        #if counter % 10 == 0:
        #    print "%i iterations" % counter
        #    print derivative
        #    print r
        if counter == 2000:
            # jesli idzie tak dlugo to pewnie i tak cos jest nie tak.
            # to niech juz beda te estymacje ktore sa.
            break
    return r

